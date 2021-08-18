import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import random from "../icons/random.svg";
import Button from "./button";

class Buttons extends Component {
  render() {
    const { clickRandomSVG } = this.props;
    return (
      <Row className="justify-content-around">
        <Button
          src={random}
          name="Shuffle"
          active={true}
          alt="Random"
          onClick={clickRandomSVG}
        />
      </Row>
    );
  }
}

export default Buttons;
