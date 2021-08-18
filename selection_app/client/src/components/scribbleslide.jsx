import React, { Component } from "react";
import SVGHandler from "./svghandler";
import Button from "./button";
import { preprocessSVG, convertCoordinates } from "../utils/svg";
import { createEmptyGraph, setDepths } from "../utils/graph";
import { cloneDeep } from "lodash";
import Row from "react-bootstrap/Row";
import Container from "react-bootstrap/Container";
import clear from "../icons/clear.svg";

class ScribbleSlide extends Component {
  constructor(props) {
    super(props);
    const graphic = preprocessSVG('<svg height="100" width="100"></svg>');
    const forest = createEmptyGraph(graphic);
    this.state = {
      id: -1,
      graphic,
      forest,
      selected: [],
      scribbleStrokes: []
    };
  }

  handleScribblePointerDown = (evt, id) => {
    this.canClear = false;
    const position = convertCoordinates(
      "demo-graphic",
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
        "demo-graphic",
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
    const { id, graphic, scribbleStrokes, selected } = this.state;
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
        selected.splice(selected.indexOf(id), 1);
      } else {
        selected.push(this.pathId);
      }
      this.setState({ selected });
      this.pathId = "undefined";
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
        fetch("/demostroke", {
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
          });
      }
    }
    scribbleStrokes.splice(currentStroke, 1);
    this.setState({ scribbleStrokes });
  };

  handleSVGPointerDown = () => {
    this.canClear = true;
  };

  handleClear = () => {
    if (this.canClear) {
      this.setState({ selected: [] });
    }
  };

  getSVGFromDB = () => {
    fetch("/demofile")
      .then(res => res.json())
      .then(item => {
        const { id, svg, forest } = item;
        setDepths(forest);
        const graphic = preprocessSVG(svg);
        this.setState({ id, graphic, forest });
      });
  };

  componentDidMount() {
    this.getSVGFromDB();
    window.addEventListener("click", this.handleClear);
  }

  componentWillUnmount() {
    window.removeEventListener("click", this.handleClear);
  }

  render() {
    const { graphic, forest } = this.state;
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <p>
            We want you to test out is the Scribble. Make rough strokes on the
            graphic. For example, trace out the arms.
          </p>
        </Row>
        <Row className="justify-content-center">
          <SVGHandler
            svgId={"demo-graphic"}
            graphic={graphic}
            forest={forest}
            scribbleStrokes={this.state.scribbleStrokes}
            selected={this.state.selected}
            onSVGPointerDown={this.handleSVGPointerDown}
            onPointerDown={this.handleScribblePointerDown}
            onPointerMove={this.handleScribblePointerMove}
            onPointerUp={this.handleScribblePointerUp}
            cx={0}
            cy={0}
            p={{ x: 0, y: 0 }}
            showSlider={false}
            onKeyPress={() => {}}
          />
        </Row>
        <Row className="justify-content-around">
          <Button
            src={clear}
            name="Clear"
            active={this.state.selected.length > 0}
            alt="Clear"
            onClick={() => this.setState({ selected: [] })}
          />
        </Row>
      </Container>
    );
  }
}

export default ScribbleSlide;
