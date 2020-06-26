import React, { Component } from "react";
import Navbar from "react-bootstrap/Navbar";
import Container from "react-bootstrap/Container";
import Controller from "./components/controller";
import logo from "./icons/logo.svg";
import confused from "./icons/confused.svg";
import Button from "./components/button";
import Help from "./components/help";

import "./App.css";

class App extends Component {
  constructor(props) {
    super(props);
    this.state = { showHelp: false };
  }

  handleHelpHide = () => {
    this.setState({ showHelp: false });
  };

  handleHelpClick = () => {
    this.setState({ showHelp: true });
  };

  render() {
    return (
      <>
        <Container id="app-container">
          <Container>
            <Navbar>
              <Navbar.Brand>
                <img
                  src={logo}
                  width="30"
                  height="30"
                  className="d-inline-block align-top"
                  alt="SVG Hierarchy"
                />
                SVG Hierarchy
              </Navbar.Brand>
              <Navbar.Collapse className="justify-content-end">
                <Button
                  src={confused}
                  name="Help"
                  alt="Help"
                  active={true}
                  onClick={this.handleHelpClick}
                ></Button>
              </Navbar.Collapse>
            </Navbar>
          </Container>
          <Controller />
        </Container>
        <Help show={this.state.showHelp} onHide={this.handleHelpHide}></Help>
      </>
    );
  }
}

export default App;
