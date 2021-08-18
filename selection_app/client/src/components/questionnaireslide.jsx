import React, { Component } from "react";
import Likert from "./likert";
import Cookies from "js-cookie";
import Row from "react-bootstrap/Row";
import Container from "react-bootstrap/Container";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import SaveToast from "./savetoast";

function capitalize(s) {
  if (typeof s !== "string") return "";
  return s.charAt(0).toUpperCase() + s.slice(1);
}

class QuestionnaireSlide extends Component {
  constructor(props) {
    super(props);
    this.state = {
      showToast: false
    };
  }

  saveComments = () => {
    const text = document.getElementById("comments").innerHTML;
    fetch("/logcomments", {
      method: "post",
      headers: {
        Accept: "application/json, text/plain, */*",
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ comments: text })
    }).then(res => this.setState({ showToast: true }));
  };

  handleShowToast = () => {
    this.setState({ showToast: false });
  }

  render() {
    const tool = Cookies.get("tool");
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <ol>
            <li>
              <Likert>I am proficient with Vector Graphics </Likert>
            </li>
            <li>
              <Likert>Task was easy</Likert>
            </li>
            <li>
              <Likert>{capitalize(tool) + " was easy to use"}</Likert>
            </li>
            <li>
              <Likert>
                {capitalize(tool) + " behaved as I expected it to"}
              </Likert>
            </li>
            <li>
              <Likert>
                {capitalize(tool) + " selected curves I wanted to select"}
              </Likert>
            </li>
            <li>
              <Likert>
                {capitalize(tool) + " selected curves I didn't want to select"}
              </Likert>
            </li>
            <li>
              <Likert>
                {capitalize(tool) + " made the task easier to complete"}
              </Likert>
            </li>
          </ol>
        </Row>
        <Row className="justify-content-center">
          <Form>
            <Form.Group controlId="comments">
              <Form.Label>Comments/Feedback</Form.Label>
              <Form.Control
                as="textarea"
                rows={3}
                name="comments"
                id="comments"
                className="small-text"
              />
            </Form.Group>
          </Form>
        </Row>
        <Row className="justify-content-center">
          <Button variant="dark" onClick={this.saveComments}>
            Submit
          </Button>
        </Row>
        <SaveToast show={this.showToast} onClose={this.handleShowToast} />
      </Container>
    );
  }
}

export default QuestionnaireSlide;
