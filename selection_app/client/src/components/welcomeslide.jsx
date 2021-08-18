import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import Container from "react-bootstrap/Container";

class WelcomeSlide extends Component {
  render() {
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <p>
            Thank you for taking part in this study. Use the arrow buttons on
            top to navigate.
          </p>
        </Row>
      </Container>
    );
  }
}

export default WelcomeSlide;
