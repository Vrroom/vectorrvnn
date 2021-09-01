import React, { Component } from "react";
import SVGHandler from "./svghandler";
import { withScribbleTool } from "./tools";
import withGraphicFetcher from "./graphicfetch";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";

const ScribbleTool = withScribbleTool(SVGHandler, "demo-graphic");

class Slide extends Component {
  render() {
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <p>
            We want you to test out is the Scribble. Make rough strokes on the
            graphic. For example, trace out the arms.
          </p>
        </Row>
        <Row>
          <Col />
          <Col> 
            <ScribbleTool svgId="demo-graphic" {...this.props} /> 
          </Col>
          <Col />
        </Row>
      </Container>
    );
  }
}

const ScribbleSlide = withGraphicFetcher(Slide, "/example");

export default ScribbleSlide;
