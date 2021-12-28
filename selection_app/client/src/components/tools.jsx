import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import { zip, cloneDeep } from "lodash";
import clear from "../icons/clear.svg";
import Button from "./button";
import { distance, subtract, convertCoordinates } from "../utils/svg";
import { ancestors } from "../utils/graph";
import { stickColor } from "../utils/palette";
import addStopPropagation from "../utils/eventModifier";

function withClickTool(Wrapped) {
  return class extends Component {
    constructor(props) {
      super(props);
      this.state = { selected: [] };
    }

    handlePointerDown = (evt, id) => {
      const selected = cloneDeep(this.state.selected);
      const isSelected = selected.includes(id);
      if (isSelected) {
        selected.splice(selected.indexOf(id), 1);
      } else {
        selected.push(id);
      }
      this.setState({ selected });
    };

    componentDidMount() {
      window.addEventListener("click", this.handleBgPointerDown);
    }

    componentWillUnmount() {
      window.removeEventListener("click", this.handleBgPointerDown);
    }

    handleBgPointerDown = () => {
      this.setState({ selected: [] });
    };

    render() {
      return (
        <>
          <Row className="justify-content-center">
            <Wrapped
              selected={this.state.selected}
              onPointerDown={this.handlePointerDown}
              onPointerMove={() => {}}
              onPointerUp={() => {}}
              toolRenderer={() => {}}
              preview={[]}
              {...this.props}
            />
          </Row>
          <Row className="justify-content-around">
            <Button
              src={clear}
              name="Clear"
              active={this.state.selected.length > 0}
              alt="Clear"
              onClick={this.handleBgPointerDown}
            />
          </Row>
        </>
      );
    }
  };
}

function withScribbleTool(Wrapped, svgId) {
  return class extends Component {
    constructor(props) {
      super(props);
      this.state = {
        selected: [],
        allStrokes: [],
        pointerDown: false,
        pathId: undefined,
      };
    }

    processStroke = (payload) => {
      fetch("/stroke", {
        method: "post",
        headers: {
          Accept: "application/json, text/plain, */*",
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      })
        .then((res) => res.json())
        .then((res) => {
          let selected = cloneDeep(this.state.selected);
          const selection = res.suggestions[0];
          selected = selected.concat(selection);
          this.setState({ selected });
        });
    };

    handlePointerDown = (evt, id) => {
      const { clientX, clientY } = evt;
      this.setState((prevState) => {
        if (!prevState.pointerDown) {
          const position = convertCoordinates(svgId, clientX, clientY);
          const { allStrokes } = this.state;
          allStrokes.push([position]);
          return { allStrokes, pointerDown: true, pathId: id };
        }
      });
    };

    handlePointerMove = (evt) => {
      const { clientX, clientY } = evt;
      this.setState((prevState) => {
        if (prevState.pointerDown) {
          const position = convertCoordinates(svgId, clientX, clientY);
          const allStrokes = cloneDeep(prevState.allStrokes);
          const current = allStrokes.length - 1;
          allStrokes[current].push(position);
          return { allStrokes };
        }
      });
    };

    handlePointerUp = (evt) => {
      this.setState((prevState) => {
        if (prevState.pointerDown) {
          const changes = {};
          const { id, graphic } = this.props;
          const { allStrokes, selected } = this.state;
          const current = allStrokes.length - 1;
          if (current < 0) return changes;
          changes.pointerDown = false;
          let currentStroke = cloneDeep(allStrokes[current]);
          if (currentStroke.length > 0 && currentStroke.length < 3 && typeof prevState.pathId !== "undefined") {
            if (selected.includes(prevState.pathId)) {
              selected.splice(selected.indexOf(id), 1);
            } else {
              selected.push(prevState.pathId);
            }
            changes.selected = selected;
            changes.pathId = undefined;
          } else if (currentStroke.length > 0) {
            const { scale } = graphic;
            currentStroke = currentStroke.map(({ x, y }) => ({ x: x / scale, y: y / scale }));
            const payload = { id, stroke: currentStroke, radius: 2 / scale };
            this.processStroke(payload);
          }
          allStrokes.splice(current, 1);
          changes.allStrokes = allStrokes;
          return changes;
        }
      });
    };

    singleStrokeRenderer = (stroke, sId) => {
      const ptPairs = zip(stroke.slice(0, -1), stroke.slice(1));
      const lines = ptPairs.map(([a, b], id) => (
        <line
          key={`line-${id}`}
          x1={a.x}
          y1={a.y}
          x2={b.x}
          y2={b.y}
          strokeWidth="2%"
          stroke="blue"
          strokeOpacity="0.4"
        />
      ));
      return <g key={`stroke-${sId}`}>{lines}</g>;
    };

    toolRenderer = () => {
      const { allStrokes } = this.state;
      const strokeElements = allStrokes.map(this.singleStrokeRenderer);
      return <g key="strokes">{strokeElements}</g>;
    };

    componentDidMount() {
      window.addEventListener("click", this.handleBgPointerDown);
    }

    componentWillUnmount() {
      window.removeEventListener("click", this.handleBgPointerDown);
    }

    handleBgPointerDown = () => {
      this.setState({ selected: [] });
    };

    render() {
      return (
        <>
          <Row className="justify-content-center">
            <Wrapped
              selected={this.state.selected}
              onPointerDown={this.handlePointerDown}
              onPointerMove={this.handlePointerMove}
              onPointerUp={this.handlePointerUp}
              toolRenderer={this.toolRenderer}
              allStrokes={this.state.allStrokes}
              preview={[]}
              {...this.props}
            />
          </Row>
          <Row className="justify-content-around">
            <Button
              src={clear}
              name="Clear"
              active={this.state.selected.length > 0}
              alt="Clear"
              onClick={this.handleBgPointerDown}
            />
          </Row>
        </>
      );
    }
  };
}

function withSliderTool(Wrapped, svgId) {
  return class extends Component {
    constructor(props) {
      super(props);
      this.state = {
        selected: [],
        oldSelected: [],
        pointerDownPt: { x: 0, y: 0 },
        pointerCurrPt: { x: 0, y: 0 },
        pointerDown: false,
        pointerJustReleased: false,
        initPathId: -1,
      };
    }

    handlePointerDown = (evt, id) => {
      const { clientX, clientY } = evt;
      this.setState((prevState) => {
        if (!prevState.pointerDown) {
          const changes = {};
          const { selected } = this.state;
          const isSelected = selected.includes(id);
          if (isSelected) {
            selected.splice(selected.indexOf(id), 1);
          } else {
            changes.oldSelected = cloneDeep(selected);
            changes.pointerDown = true;
            changes.initPathId = id;
            const point = convertCoordinates(svgId, clientX, clientY);
            changes.pointerDownPt = point;
            changes.pointerCurrPt = point;
            selected.push(id);
          }
          changes.selected = selected;
          return changes;
        }
      });
    };

    handlePointerMove = (evt) => {
      const { clientX, clientY } = evt;
      this.setState((prevState) => {
        if (prevState.pointerDown) {
          const { graphic, forest } = this.props;
          const { oldSelected, pointerDownPt, initPathId } = this.state;
          const dim = graphic.svg.properties.height;
          const point = convertCoordinates(svgId, clientX, clientY);
          const dist = distance(point, pointerDownPt);
          const anc = ancestors(forest, initPathId);
          const frac = (2 * dist) / dim;
          const id = Math.min(anc.length - 1, Math.floor(frac * anc.length));
          const newNode = anc[id];
          const newSelected = forest.nodes[newNode].paths;
          return { selected: oldSelected.concat(newSelected), pointerCurrPt: point };
        }
      });
    };

    handlePointerUp = (evt) => {
      this.setState((prevState) => {
        if (prevState.pointerDown) {
          return {
            pointerDown: false,
            initPathId: -1,
            pointerDownPt: { x: 0, y: 0 },
            pointerCurrPt: { x: 0, y: 0 },
            pointerJustReleased: true,
          };
        }
      });
    };

    componentDidMount() {
      window.addEventListener("click", this.handleBgPointerDown);
    }

    componentWillUnmount() {
      window.removeEventListener("click", this.handleBgPointerDown);
    }

    handleBgPointerDown = () => {
      this.setState((prevState) => {
        if (!prevState.pointerJustReleased) {
          return { selected: [], oldSelected: [] };
        }
        return { pointerJustReleased: false };
      });
    };

    toolRenderer = () => {
      const { pointerDown, pointerDownPt, pointerCurrPt } = this.state;
      if (pointerDown) {
        const diff = subtract(pointerCurrPt, pointerDownPt);
        const otherEnd = subtract(pointerDownPt, diff);
        const r = distance(pointerCurrPt, otherEnd) / 2 + 1e-2;
        const perp = { x: (-2 * diff.y) / r, y: (2 * diff.x) / r };
        return (
          <g id="slider" stroke={stickColor} strokeWidth="2%" fill="transparent">
            <line x1={pointerCurrPt.x} y1={pointerCurrPt.y} x2={otherEnd.x} y2={otherEnd.y} />
            <line
              x1={pointerCurrPt.x - perp.x}
              y1={pointerCurrPt.y - perp.y}
              x2={pointerCurrPt.x + perp.x}
              y2={pointerCurrPt.y + perp.y}
            />
            <line x1={otherEnd.x - perp.x} y1={otherEnd.y - perp.y} x2={otherEnd.x + perp.x} y2={otherEnd.y + perp.y} />
          </g>
        );
      } else {
        return <g id="slider" />;
      }
    };

    render() {
      return (
        <>
          <Row className="justify-content-center">
            <Wrapped
              selected={this.state.selected}
              onPointerDown={this.handlePointerDown}
              onPointerMove={this.handlePointerMove}
              onPointerUp={this.handlePointerUp}
              toolRenderer={this.toolRenderer}
              preview={[]}
              {...this.props}
            />
          </Row>
          <Row className="justify-content-around">
            <Button
              src={clear}
              name="Clear"
              active={this.state.selected.length > 0}
              alt="Clear"
              onClick={(evt) => this.setState({ selected: [] })}
            />
          </Row>
        </>
      );
    }
  };
}

function withToggleTool(Wrapped, svgId) {
  return class extends Component {
    constructor(props) {
      super(props);
      this.state = {
        selected: [],
        pointerDown: false,
        pointerJustReleased: false,
        initPathId: -1,
        hierarchyLevel: 0,
        oldSelected: [],
        hoverPlus: false,
        hoverMinus: false,
      };
    }

    handlePointerDown = (evt, id) => {
      this.setState((prevState) => {
        if (!prevState.pointerDown) {
          const changes = {};
          const { selected } = this.state;
          const isSelected = selected.includes(id);
          if (isSelected) {
            selected.splice(selected.indexOf(id), 1);
          } else {
            changes.pointerDown = true;
            changes.initPathId = id;
            changes.hierarchyLevel = 0;
            changes.oldSelected = cloneDeep(selected);
            selected.push(id);
          }
          changes.selected = selected;
          return changes;
        }
      });
    };

    handlePointerUp = (evt) => {
      this.setState((prevState) => {
        if (prevState.pointerDown) {
          return { pointerDown: false, pointerJustReleased: true };
        }
      });
    };

    componentDidMount() {
      window.addEventListener("click", this.handleBgPointerDown);
    }

    componentWillUnmount() {
      window.removeEventListener("click", this.handleBgPointerDown);
    }

    handleBgPointerDown = () => {
      this.setState((prevState) => {
        if (!prevState.pointerJustReleased) {
          return { selected: [], oldSelected: [] };
        }
        return { pointerJustReleased: false };
      });
    };

    handleToggleClick = (delta) => {
      this.setState((prevState) => {
        const { forest } = this.props;
        const { hierarchyLevel, initPathId, oldSelected } = prevState;
        const anc = ancestors(forest, initPathId);
        const id = hierarchyLevel + delta;
        if (id >= 0 && id < anc.length) {
          const newSelected = forest.nodes[anc[id]].paths;
          const selected = newSelected.concat(oldSelected);
          return { hierarchyLevel: id, selected };
        }
      });
    };

    handlePlusPointerEnter = (evt) => {
      this.setState({ hoverPlus: true });
    };

    handlePlusPointerLeave = (evt) => {
      this.setState({ hoverPlus: false });
    };

    handleMinusPointerEnter = (evt) => {
      this.setState({ hoverMinus: true });
    };

    handleMinusPointerLeave = (evt) => {
      this.setState({ hoverMinus: false });
    };

    toolRenderer = () => {
      const { pointerJustReleased } = this.state;
      if (pointerJustReleased) {
        return (
          <g stroke="#000" transform={`translate(80 20)`}>
            <rect
              x="-5"
              y="-10"
              width="10"
              height="20"
              rx=".5"
              ry=".5"
              fillOpacity="0"
              strokeLinecap="round"
              strokeWidth="2"
            />
            <g fill="none">
              <path d="m-5 0h10" strokeWidth="2" />
              <g strokeWidth="1px">
                <path d="m0-7v4" />
                <path d="m-2-5h4" />
                <path d="m-2 5h4" />
              </g>
            </g>
            <rect
              onClick={addStopPropagation((evt) => this.handleToggleClick(-1))}
              onPointerEnter={this.handleMinusPointerEnter}
              onPointerLeave={this.handleMinusPointerLeave}
              id="toggle-button"
              x="-5"
              width="10"
              height="10"
              rx=".5"
              ry=".5"
              fillOpacity={this.state.hoverMinus ? "0.1" : ".25"}
              strokeLinecap="round"
              strokeWidth="2"
            />
            <rect
              onClick={addStopPropagation((evt) => this.handleToggleClick(1))}
              onPointerEnter={this.handlePlusPointerEnter}
              onPointerLeave={this.handlePlusPointerLeave}
              id="toggle-button"
              x="-5"
              y="-10"
              width="10"
              height="10"
              rx=".5"
              ry=".5"
              fillOpacity={this.state.hoverPlus ? "0.1" : ".25"}
              strokeLinecap="round"
              strokeWidth="2"
            />
          </g>
        );
      } else {
        return <g id="toggle" />;
      }
    };

    getPreview = () => {
      const { initPathId, hoverPlus, hierarchyLevel } = this.state;
      const { forest } = this.props;
      let preview = [];
      if (initPathId >= 0 && hoverPlus) {
        const anc = ancestors(forest, initPathId);
        const id = hierarchyLevel + 1;
        if (id < anc.length) { 
          const node = anc[id];
          preview = forest.nodes[node].paths;
        }
      }
      return preview;
    }

    render() {
      const { selected, pointerJustReleased } = this.state;
      return (
        <>
          <Row className="justify-content-center">
            <Wrapped
              selected={selected}
              onPointerDown={this.handlePointerDown}
              onPointerMove={() => {}}
              onPointerUp={this.handlePointerUp}
              toolRenderer={this.toolRenderer}
              pointerJustReleased={pointerJustReleased}
              preview={this.getPreview()}
              {...this.props}
            />
          </Row>
          <Row className="justify-content-around">
            <Button
              src={clear}
              name="Clear"
              active={this.state.selected.length > 0}
              alt="Clear"
              onClick={(evt) => this.setState({ selected: [] })}
            />
          </Row>
        </>
      );
    }
  };
}

export { withClickTool, withScribbleTool, withSliderTool, withToggleTool };
