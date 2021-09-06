import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import SVGHandler from "./svghandler";
import Cookies from "js-cookie";
import { withSliderTool, withScribbleTool } from "./tools";
import withGraphicFetcher from "./graphicfetch";
import next from "../icons/next.svg";
import Button from "./button";
import { targetGraphic } from "../utils/svg";

function capitalize(s) {
  if (typeof s !== "string") return "";
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const tool = Cookies.get("tool");
const toolMaker = {
  slider: withSliderTool,
  scribble: withScribbleTool,
};

const Tool = toolMaker[tool](SVGHandler, "graphic");

const MAX_TASKS = 10;

class Slide extends Component {
  constructor(props) {
    super(props);
    this.state = { graphicNum: 1 };
  }

  handleNextClick = () => {
    this.props.fetcher();
    this.setState((prevState) => ({ graphicNum: prevState.graphicNum + 1 }));
  };

  render() {
    const { graphic, target } = this.props;
    const bwGraphic = targetGraphic(graphic, target);
    return (
      <Container>
        <Row>
          <Col>
            <Row className="normal-text justify-content-center">{`Select objects with ${capitalize(tool)}`}</Row>
          </Col>
          <Col>
            <Row className="justify-content-center normal-text">Target Selection (in red):</Row>
          </Col>
        </Row>
        <Row>
          <Col>
            <Tool svgId="graphic" {...this.props} />
          </Col>
          <Col>
            <SVGHandler
              svgId="task"
              selected={[]}
              onPointerDown={() => {}}
              onPointerMove={() => {}}
              onPointerUp={() => {}}
              toolRenderer={() => {}}
              preview={[]}
              graphic={bwGraphic}
            />
          </Col>
        </Row>
        <Row className="justify-content-around">
          <Button
            src={next}
            name="Next"
            active={this.state.graphicNum < MAX_TASKS}
            alt="Next"
            onClick={this.handleNextClick}
          />
          {`${this.state.graphicNum}/${MAX_TASKS}`}
        </Row>
      </Container>
    );
  }
}

const TaskSlide = withGraphicFetcher(Slide, "/task");

export default TaskSlide;
