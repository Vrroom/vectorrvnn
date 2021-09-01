import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import { zip, cloneDeep } from "lodash";
import clear from "../icons/clear.svg";
import Button from "./button";
import { distance, subtract, convertCoordinates } from "../utils/svg";
import { ancestors } from "../utils/graph";
import { stickColor } from "../utils/palette";

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
          const frac = 2 * dist / dim;
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

export { withClickTool, withScribbleTool, withSliderTool };
