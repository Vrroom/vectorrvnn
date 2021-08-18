import React, { Component } from "react";
import SVGHandler from "./svghandler";
import Button from "./button";
import { preprocessSVG } from "../utils/svg";
import { createEmptyGraph, setDepths } from "../utils/graph";
import { cloneDeep } from "lodash";
import Row from "react-bootstrap/Row";
import Container from "react-bootstrap/Container";
import clear from "../icons/clear.svg";

class VectorGraphicSlide extends Component {
  constructor(props) {
    super(props);
    const graphic = preprocessSVG('<svg height="100" width="100"></svg>');
    const forest = createEmptyGraph(graphic);
    this.state = { graphic, forest, selected: [] };
  }

  getSVGFromDB = () => {
    fetch("/db")
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

  handlePointerDown = (evt, id) => {
    this.canClear = false;
    const selected = cloneDeep(this.state.selected);
    const isSelected = selected.includes(id);
    if (isSelected) {
      selected.splice(selected.indexOf(id), 1);
      this.setState({ selected });
    } else {
      selected.push(id);
      this.setState({ selected });
    }
  };

  handleClear = () => {
    if (this.canClear) {
      this.setState({ selected: [] });
    }
  };

  render() {
    const { graphic, forest } = this.state;
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <p>
            Vector Graphics are resolution-independent images. Such graphics are
            made up of many individual objects. Try clicking on some object in
            the graphic below. The clicked object gets selected (highlighted in
            blue). Click on the object again or the clear button to de-select.
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
            onPointerDown={this.handlePointerDown}
            onPointerMove={() => {}}
            onPointerUp={() => {}}
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
            onClick={() => {
              this.setState({ selected: [] });
            }}
          />
        </Row>
      </Container>
    );
  }
}

export default VectorGraphicSlide;
