import React, { Component } from "react";
import SVGHandler from "./svghandler";
import { withToggleTool } from "./tools";
import withGraphicFetcher from "./graphicfetch";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";

const ToggleTool = withToggleTool(SVGHandler, "demo-graphic");

class Slide extends Component {
  render() {
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <p>
            We want you to test out the Toggle. Click on an object and use the expand contract icons to select.
          </p>
        </Row>
        <Row>
          <Col />
          <Col> 
            <ToggleTool svgId="demo-graphic" {...this.props} /> 
          </Col>
          <Col />
        </Row>
      </Container>
    );
  }
}

const ToggleSlide = withGraphicFetcher(Slide, "/example");

export default ToggleSlide;

