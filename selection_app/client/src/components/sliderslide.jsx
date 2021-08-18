import React, { Component } from "react";
import SVGHandler from "./svghandler";
import Button from "./button";
import { preprocessSVG, convertCoordinates, distance } from "../utils/svg";
import { createEmptyGraph, setDepths } from "../utils/graph";
import { cloneDeep } from "lodash";
import Row from "react-bootstrap/Row";
import Container from "react-bootstrap/Container";
import clear from "../icons/clear.svg";

class SliderSlide extends Component {
  constructor(props) {
    super(props);
    const graphic = preprocessSVG('<svg height="100" width="100"></svg>');
    const forest = createEmptyGraph(graphic);
    this.state = {
      graphic,
      forest,
      selected: [],
      p: { x: 0, y: 0 },
      cx: 0,
      cy: 0,
      sliderPathToRoot: [],
      showSlider: false
    };
  }

  handleSliderPointerDown = (evt, id) => {
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
      const point = convertCoordinates(
        "demo-graphic",
        evt.clientX,
        evt.clientY
      );
      const forest = cloneDeep(this.state.forest);
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
        sliderPathToRoot,
        showSlider: true,
        selected
      });
    }
  };

  handleSliderPointerMove = evt => {
    if (this.sliderPointerDown) {
      const { cx, cy, sliderPathToRoot, graphic, forest } = this.state;
      let selected = cloneDeep(this.state.selected);
      const d = graphic.svg.properties.height;
      const p = convertCoordinates("demo-graphic", evt.clientX, evt.clientY);
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
    this.setState({ showSlider: false, sliderPathToRoot: [] });
    this.sliderPointerDown = false;
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
        const { svg, forest } = item;
        setDepths(forest);
        const graphic = preprocessSVG(svg);
        this.setState({ graphic, forest });
      });
  };

  handleSVGPointerDown = () => {
    this.canClear = true;
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
            We want you to test out the Slider. Click on an object and drag
            in any direction. Initially, closely related objects are added to
            current selection. As the length of the slider increases, more
            objects are added. Eventually all objects are selected.
          </p>
        </Row>
        <Row className="justify-content-center">
          <SVGHandler
            svgId={"demo-graphic"}
            graphic={graphic}
            forest={forest}
            scribbleStrokes={[[]]}
            selected={this.state.selected}
            onSVGPointerDown={this.handleSVGPointerDown}
            onPointerDown={this.handleSliderPointerDown}
            onPointerMove={this.handleSliderPointerMove}
            onPointerUp={this.handleSliderPointerUp}
            cx={this.state.cx}
            cy={this.state.cy}
            p={this.state.p}
            showSlider={this.state.showSlider}
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

export default SliderSlide;
