import React, { Component } from "react";
import SVGHandler from "./svghandler";
import { withClickTool } from "./tools";
import withGraphicFetcher from "./graphicfetch";
import Container from "react-bootstrap/Container";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";

const ClickTool = withClickTool(SVGHandler);

class Slide extends Component {
  render() {
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <p>
            Vector Graphics are resolution-independent images. Such graphics are made up of many individual objects. Try
            clicking on some object in the graphic below. The clicked object gets selected (highlighted in blue). Click
            on the object again or the clear button to de-select.
          </p>
        </Row>
        <Row>
          <Col />
          <Col> 
            <ClickTool svgId="demo-graphic" {...this.props} /> 
          </Col>
          <Col />
        </Row>
      </Container>
    );
  }
}

const VectorGraphicSlide = withGraphicFetcher(Slide, "/example");

export default VectorGraphicSlide;
