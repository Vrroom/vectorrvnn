import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import ClickableText from "./clickabletext";

class Likert extends Component {
  constructor(props) {
    super(props);
    this.state = { active: true, selectedId: undefined };
    this.options = [
      "Strongly Disagree",
      "Disagree",
      "Neutral",
      "Agree",
      "Strongly Agree"
    ];
  }

  handleClick = score => {
    // Send question and score to database.
    if (this.state.active) {
      const data = {
        question: this.props.children,
        score
      };
      fetch("/surveyquestion", {
        method: "post",
        headers: {
          Accept: "application/json, text/plain, */*",
          "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
      });
      this.setState({ active: false, selectedId: score });
    }
  };

  render() {
    return (
      <Container className="normal-text">
        <Row>{this.props.children}</Row>
        <Row className="justify-content-around">
          {this.options.map((option, id) => {
            return (
              <Col key={`option-${id}`}>
                <ClickableText
                  onClick={() => this.handleClick(id + 1)}
                  chosen={this.state.selectedId === id + 1}
                  active={this.state.active}
                >
                  {option}
                </ClickableText>
              </Col>
            );
          })}
        </Row>
      </Container>
    );
  }
}

export default Likert;
