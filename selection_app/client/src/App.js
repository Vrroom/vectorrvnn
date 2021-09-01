import React, { Component } from "react";
import Container from "react-bootstrap/Container";
import Button from "./components/button";
import Row from "react-bootstrap/Row";
import leftarrow from "./icons/leftarrow.svg";
import rightarrow from "./icons/rightarrow.svg";
import TaskSlide from "./components/taskslide";
import VectorGraphicSlide from "./components/vectorgraphicslide";
import SliderSlide from "./components/sliderslide";
import ScribbleSlide from "./components/scribbleslide";
import WelcomeSlide from "./components/welcomeslide";
import QuestionnaireSlide from "./components/questionnaireslide";
import AboutSlide from "./components/aboutslide";
import ByeSlide from "./components/byeslide";
import Cookies from "js-cookie";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = {
      slideNo: 0
    };
  }

  navigateSlide = dir => {
    const { slideNo } = this.state;
    this.setState({ slideNo: (slideNo + 7 + dir) % 7 });
  };

  slideClick = slideNo => {
    this.setState({ slideNo });
  };

  getSlide = slideNo => {
    const tool = Cookies.get("tool");
    switch (slideNo) {
      case 0:
        return {
          title: "Welcome!",
          content: <WelcomeSlide onClick={this.slideClick} />
        };
      case 1:
        return {
          title: "About",
          content: <AboutSlide />
        };
      case 2:
        return {
          title: "Vector Graphics",
          content: <VectorGraphicSlide />
        };
      case 3:
        if (tool === "slider") {
          return {
            title: "Selection Tool",
            content: <SliderSlide />
          };
        } else {
          return {
            title: "Selection Tool",
            content: <ScribbleSlide />
          };
        }
      case 4:
        return {
          title: "Task",
          content: <TaskSlide />
        };
      case 5:
        return {
          title: "Questionnaire",
          content: <QuestionnaireSlide />
        };
      case 6: 
        return {
          title: "Bye Bye!",
          content: <ByeSlide />
        };
      default:
        return { title: "", content: null };
    }
  };

  userStudyUI = () => {
    const { slideNo } = this.state;
    const slide = this.getSlide(slideNo);
    return (
      <Container id="app-container">
        <Row className="justify-content-around">
          <Button
            src={leftarrow}
            name="Previous"
            active={slideNo > 0}
            alt="Previous"
            onClick={() => this.navigateSlide(-1)}
          />
          {slide.title}
          <Button
            src={rightarrow}
            name="Next"
            active={slideNo < 6}
            alt="Next"
            onClick={() => this.navigateSlide(1)}
          />
        </Row>
        {slide.content}
      </Container>
    );
  };

  render() {
    return this.userStudyUI();
  }
}

export default App;
