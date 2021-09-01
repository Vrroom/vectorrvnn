import React, { Component } from "react";
import SVGHandler from "./svghandler";
import { withSliderTool } from "./tools";
import withGraphicFetcher from "./graphicfetch";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";

const SliderTool = withSliderTool(SVGHandler, "demo-graphic");

class Slide extends Component {
  render() {
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <p>
            We want you to test out the Slider. Click on an object and drag in any direction. Initially, closely related
            objects are added to current selection. As the length of the slider increases, more objects are added.
            Eventually all objects are selected.
          </p>
        </Row>
        <Row>
          <Col />
          <Col> 
            <SliderTool svgId="demo-graphic" {...this.props} /> 
          </Col>
          <Col />
        </Row>
      </Container>
    );
  }
}

const SliderSlide = withGraphicFetcher(Slide, "/example");

export default SliderSlide;
