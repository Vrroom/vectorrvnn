import React, { Component } from "react";
import io from "socket.io-client";
import Dropdown from "react-bootstrap/Dropdown";
import DropdownButton from "react-bootstrap/DropdownButton";
import ListGroup from "react-bootstrap/ListGroup";
import Button from "react-bootstrap/Button";
import InputGroup from "react-bootstrap/InputGroup";
import FormControl from "react-bootstrap/FormControl";

class Connections extends Component {
  constructor(props) {
    super(props);
    this.nameInput = React.createRef();
    this.state = { hasNickName: false };
  }

  handleCreateNickName = () => {
    this.setState({ hasNickName: true });
    this.username = this.nameInput.current.value;
    this.props.socket.emit("add-user", this.nameInput.current.value);
  };

  getNickName = () => {
    return (
      <InputGroup className="mb-3">
        <FormControl
          ref={this.nameInput}
          placeholder="Nick Name"
          aria-label="Nick Name"
          aria-describedby="basic-addon2"
        />
        <InputGroup.Append>
          <Button
            variant="outline-secondary"
            onClick={this.handleCreateNickName}
          >
            Submit
          </Button>
        </InputGroup.Append>
      </InputGroup>
    );
  };

  getOnline = () => {
    const { online, handleSend, handleCompare } = this.props;
    if (online.length === 1) {
      return <p>You are the only one here</p>
    }
    const copy = [...online];
    copy.splice(copy.indexOf(this.username), 1);
    const people = copy.map((username, i) => {
      return (
        <ListGroup.Item key={`player-${i}`}>
          <DropdownButton
            variant="outline-secondary"
            key={`dropdown-${i}`}
            title={username}
          >
            <Dropdown.Item
              key={`send-${i}`}
              onClick={() => handleSend(username)}
            >
              Send
            </Dropdown.Item>
            <Dropdown.Item
              key={`compare-${i}`}
              onClick={() => handleCompare(username)}
            >
              Compare
            </Dropdown.Item>
          </DropdownButton>
        </ListGroup.Item>
      );
    });
    return <ListGroup>{people}</ListGroup>;
  };

  render() {
    if (this.state.hasNickName) {
      return this.getOnline();
    } else {
      return this.getNickName();
    }
  }
}

export default Connections;
