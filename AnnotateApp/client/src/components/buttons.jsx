import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import save from "../icons/save.svg";
import random from "../icons/random.svg";
import group from "../icons/group.svg";
import clear from "../icons/clear.svg";
import Button from "./button";

class Buttons extends Component {
  render() {
    const {
      clickRandomSVG,
      clickGroup,
      clickClear,
      clickSave
    } = this.props;
    return (
      <Row className="justify-content-around">
        <Button 
          src={random}
          name="Get Random SVG"
          active={true}
          alt="Random"
          onClick={clickRandomSVG}
        />
        <Button 
          src={group}
          name="Group"
          active={true}
          alt="Group"
          onClick={clickGroup}
        />
        <Button 
          src={clear}
          name="Clear Selection"
          active={true}
          alt="Clear"
          onClick={clickClear}
        />
        <Button 
          src={save}
          name="Save Graph"
          active={true}
          alt="Save"
          toast={true}
          onClick={clickSave}
        />
      </Row>
    );
  }
}

export default Buttons;
