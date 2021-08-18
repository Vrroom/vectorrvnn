import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import SVGHandler from "./svghandler";
import Button from "./button";
import clear from "../icons/clear.svg";
import next from "../icons/next.svg";
import Cookies from "js-cookie";
import {
  preprocessSVG,
  convertCoordinates,
  distance,
  initialCanvasTransform
} from "../utils/svg";
import { cloneDeep } from "lodash";
import { createEmptyGraph, setDepths } from "../utils/graph";

function capitalize(s) {
  if (typeof s !== "string") return "";
  return s.charAt(0).toUpperCase() + s.slice(1);
}

class Controller extends Component {
  /*
   * Set the initial state of the component.
   *
   * This is just a formality because the state
   * would be over-written when the component mounts
   * because there, we can do an AJAX call to retrieve
   * an SVG from the server.
   *
   * Here we use a placeholder SVG string.
   */
  constructor(props) {
    super(props);
    const graphic = preprocessSVG('<svg height="100" width="100"></svg>');
    const tool = Cookies.get("tool");
    const backend = Cookies.get("backend");
    console.log(tool, backend);
    this.state = {
      sliderPathToRoot: [],
      selected: [],
      cx: 0,
      cy: 0,
      showSlider: false,
      p: { x: 0, y: 0 },
      elementsOnCanvas: [],
      canvasSelected: [],
      mousePosition: { x: 0, y: 0 },
      showCopyToast: false,
      id: -1,
      graphic,
      forest: createEmptyGraph(graphic),
      scribbleStrokes: [],
      toggleSwitchValue: tool === "scribble",
      selectedPaletteId: 0,
      selectedByMethod: [],
      beforeSelected: [],
      nGraphics: 0,
      im: undefined
    };
    this.clipboard = undefined;
    this.canClear = true;
    this.cornerPointerDown = false;
    this.drawingCanvasPointerDown = false;
    this.scribblePointerDown = false;
    this.sliderPointerDown = false;
    this.cid = undefined;
    this.eid = undefined;
    this.pathId = undefined;
  }

  fetchGraphicFromDB = () => {
    fetch("/taskgraphic", {
      method: "post",
      headers: {
        Accept: "application/json, text/plain, */*",
        "Content-Type": "application/json"
      }
    })
      .then(res => res.json())
      .then(item => {
        const { svg, id, forest, im } = item;
        setDepths(forest);
        const graphic = preprocessSVG(svg);
        const { nGraphics } = this.state;
        this.setState({ graphic, id, forest, im, nGraphics: nGraphics + 1 });
      });
  };

  pasteSelectedItems = evt => {
    if (typeof this.clipboard === "undefined") {
      return;
    }
    const elementsOnCanvas = cloneDeep(this.state.elementsOnCanvas);
    const { graphic, pathIdx } = this.clipboard;
    elementsOnCanvas.push({
      transforms: initialCanvasTransform(graphic, pathIdx),
      ...this.clipboard
    });
    this.setState({
      elementsOnCanvas
    });
    const data = {
      task: 2,
      selectedByMethod: this.state.selectedByMethod
    };
    fetch("/clicklog", {
      method: "post",
      headers: {
        Accept: "application/json, text/plain, */*",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    }).then(res => {
      this.setState({ selectedByMethod: [] });
    });
  };

  handleWindowPointerDown = () => {
    this.canClear = true;
  };

  handleSVGPointerDown = () => {
    this.canClear = true;
  };

  /*
   * When the component mounts, add an event listener for
   * click. Any click which isn't caught by a child element
   * of window will be caught here and whatever has been
   * selected by the user would be cleared
   *
   * Also fetch a new graphic from the database.
   */
  componentDidMount() {
    // should be mounted on the app component, not here.
    window.addEventListener("pointerdown", this.handleWindowPointerDown);
    window.addEventListener("keydown", this.handleKeyDown);
    window.addEventListener("click", this.handleClear);
    this.fetchGraphicFromDB();
  }

  /*
   * When the component unmounts, remove the click
   * event listener.
   */
  componentWillUnmount() {
    window.removeEventListener("pointerdown", this.handleWindowPointerDown);
    window.removeEventListener("keydown", this.handleKeyDown);
    window.removeEventListener("click", this.handleClear);
    const data = {
      id: this.state.id,
      selectedByMethod: this.state.selectedByMethod
    };
    fetch("/clicklog", {
      method: "post",
      headers: {
        Accept: "application/json, text/plain, */*",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    }).then(res => {
      this.setState({ selectedByMethod: [] });
    });
  }

  handleSliderPointerDown = (evt, id) => {
    this.setState((state, props) => {
      return { beforeSelected: cloneDeep(state.selected) };
    });
    this.canClear = false;
    const selected = cloneDeep(this.state.selected);
    const isSelected = selected.includes(id);
    this.sliderPointerDown = true;
    if (isSelected) {
      selected.splice(selected.indexOf(id), 1);
      this.setState({ selected });
    } else {
      this.newItemInsertId = selected.length;
      selected.push(id);
      const point = convertCoordinates("svg-element", evt.clientX, evt.clientY);
      const { forest } = this.state;
      const sliderPathToRoot = cloneDeep(this.state.sliderPathToRoot);
      sliderPathToRoot.push(id);
      let parent = forest.nodes[id].parent;
      while (typeof parent !== "undefined") {
        sliderPathToRoot.push(parent);
        parent = forest.nodes[parent].parent;
      }
      this.setState({
        p: point,
        cx: point.x,
        cy: point.y,
        showSlider: true,
        sliderPathToRoot,
        selected
      });
    }
  };

  handleSliderPointerMove = evt => {
    if (this.sliderPointerDown) {
      const { cx, cy, sliderPathToRoot, graphic, forest } = this.state;
      let selected = cloneDeep(this.state.selected);
      const d = graphic.svg.properties.height;
      const p = convertCoordinates("svg-element", evt.clientX, evt.clientY);
      const r = distance(p, { x: cx, y: cy });
      let id = Math.floor(((2 * r) / d) * sliderPathToRoot.length);
      id = Math.min(id, sliderPathToRoot.length - 1);
      const nodeId = sliderPathToRoot[id];
      selected.splice(
        this.newItemInsertId,
        selected.length - this.newItemInsertId
      );
      selected = selected.concat(forest.nodes[nodeId].paths);
      this.setState({ p, selected });
    }
  };

  handleSliderPointerUp = evt => {
    this.setState((state, props) => {
      const selectedByMethod = cloneDeep(state.selectedByMethod);
      const addDiff = state.selected.filter(
        i => !state.beforeSelected.includes(i)
      );
      const subDiff = state.beforeSelected.filter(
        i => !state.selected.includes(i)
      );
      selectedByMethod.push({ method: "slider", addDiff, subDiff });
      return { selectedByMethod, beforeSelected: [] };
    });
    this.setState({ showSlider: false, sliderPathToRoot: [] });
    this.sliderPointerDown = false;
  };

  /*
   * Clear the selections.
   *
   * Whenever any useless part of the window
   * is clicked, de-select all the selected paths.
   * This is what happens in a lot of graphics
   * editors.
   */
  handleClear = () => {
    if (this.canClear) {
      this.setState({
        selected: [],
        canvasSelected: [],
        beforeSelected: [],
        selectedByMethod: []
      });
    }
  };

  handleScribblePointerDown = (evt, id) => {
    this.setState((state, props) => {
      return { beforeSelected: cloneDeep(state.selected) };
    });
    this.canClear = false;
    const position = convertCoordinates(
      "svg-element",
      evt.clientX,
      evt.clientY
    );
    this.pathId = id;
    this.scribblePointerDown = true;
    const { scribbleStrokes } = this.state;
    scribbleStrokes.push([position]);
    this.setState({ scribbleStrokes });
  };

  handleScribblePointerMove = evt => {
    if (this.scribblePointerDown) {
      const position = convertCoordinates(
        "svg-element",
        evt.clientX,
        evt.clientY
      );
      const scribbleStrokes = cloneDeep(this.state.scribbleStrokes);
      const currentStroke = scribbleStrokes.length - 1;
      scribbleStrokes[currentStroke].push(position);
      this.setState({ scribbleStrokes });
    }
  };

  handleScribblePointerUp = evt => {
    this.scribblePointerDown = false;
    const { id, graphic, scribbleStrokes } = this.state;
    const selected = cloneDeep(this.state.selected);
    const currentStroke = scribbleStrokes.length - 1;
    if (scribbleStrokes.length === 0) {
      this.canClear = true;
      return;
    }
    const { scale } = graphic;
    const currentScribbleStroke = cloneDeep(scribbleStrokes[currentStroke]);
    if (
      currentScribbleStroke.length > 0 &&
      currentScribbleStroke.length < 3 &&
      typeof this.pathId !== "undefined"
    ) {
      if (selected.includes(this.pathId)) {
        selected.splice(selected.indexOf(this.pathId), 1);
      } else {
        selected.push(this.pathId);
      }
      this.setState({ selected });
      this.setState((state, props) => {
        const selectedByMethod = cloneDeep(state.selectedByMethod);
        const addDiff = state.selected.filter(
          i => !state.beforeSelected.includes(i)
        );
        const subDiff = state.beforeSelected.filter(
          i => !state.selected.includes(i)
        );
        selectedByMethod.push({ method: "scribble", addDiff, subDiff });
        return { selectedByMethod, beforeSelected: [] };
      });
      this.pathId = undefined;
    } else {
      for (let i = 0; i < currentScribbleStroke.length; i++) {
        currentScribbleStroke[i].x /= scale;
        currentScribbleStroke[i].y /= scale;
      }
      const data = {
        id,
        stroke: currentScribbleStroke,
        radius: 2 / scale
      };
      if (data.stroke.length > 0) {
        fetch("/stroke", {
          method: "post",
          headers: {
            Accept: "application/json, text/plain, */*",
            "Content-Type": "application/json"
          },
          body: JSON.stringify(data)
        })
          .then(res => res.json())
          .then(res => {
            let selected = cloneDeep(this.state.selected);
            const selection = res.suggestions[0];
            selected = selected.concat(selection);
            this.setState({ selected });
            this.setState((state, props) => {
              const selectedByMethod = cloneDeep(state.selectedByMethod);
              const addDiff = state.selected.filter(
                i => !state.beforeSelected.includes(i)
              );
              const subDiff = state.beforeSelected.filter(
                i => !state.selected.includes(i)
              );
              selectedByMethod.push({ method: "scribble", addDiff, subDiff });
              return { selectedByMethod, beforeSelected: [] };
            });
          });
      }
    }
    scribbleStrokes.splice(currentStroke, 1);
    this.setState({ scribbleStrokes });
  };

  handleNextClick = () => {
    this.fetchGraphicFromDB();
    const data = {
      id: this.state.id,
      selectedByMethod: this.state.selectedByMethod
    };
    fetch("/clicklog", {
      method: "post",
      headers: {
        Accept: "application/json, text/plain, */*",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    }).then(res => {
      this.setState({ selectedByMethod: [] });
    });
  }

  render() {
    const { graphic, forest } = this.state;
    const tool = Cookies.get("tool");
    return (
      <Container>
        <Row>
          <Col>
            <Row className="normal-text justify-content-center">
              {`Select objects with ${capitalize(tool)}`}
            </Row>
            <Row>
              <SVGHandler
                graphic={graphic}
                forest={forest}
                scribbleStrokes={this.state.scribbleStrokes}
                selected={this.state.selected}
                onSVGPointerDown={this.handleSVGPointerDown}
                onPointerDown={
                  this.state.toggleSwitchValue
                    ? this.handleScribblePointerDown
                    : this.handleSliderPointerDown
                }
                onPointerMove={
                  this.state.toggleSwitchValue
                    ? this.handleScribblePointerMove
                    : this.handleSliderPointerMove
                }
                onPointerUp={
                  this.state.toggleSwitchValue
                    ? this.handleScribblePointerUp
                    : this.handleSliderPointerUp
                }
                cx={this.state.cx}
                cy={this.state.cy}
                p={this.state.p}
                showSlider={this.state.showSlider}
              />
            </Row>
          </Col>
          <Col>
            <Row className="justify-content-center normal-text">
              Target Selection (in red):
            </Row>
            <Row>
              <img
                src={this.state.im}
                width="100%"
                height="auto"
                alt="Selection Target"
              />
            </Row>
          </Col>
        </Row>
        <Row className="justify-content-around">
          <Button
            src={clear}
            name="Clear"
            active={this.state.selected.length > 0}
            alt="Clear"
            onClick={() => this.setState({ selected: [] })}
          />
          <Button
            src={next}
            name="Next"
            active={this.state.nGraphics < 5}
            alt="Next"
            onClick={this.handleNextClick}
          />
          {`${this.state.nGraphics}/5`}
        </Row>
      </Container>
    );
  }
}

export default Controller;
