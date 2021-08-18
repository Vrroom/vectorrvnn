import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import Container from "react-bootstrap/Container";

class AboutSlide extends Component {
  render() {
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <p>
            We aim to compare 4 tools for selecting objects in Vector Graphics.
            We ask you to use one of them to complete some tasks and then tell
            us about your experience.
          </p>
        </Row>
      </Container>
    );
  }
}

export default AboutSlide;
