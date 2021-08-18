import React, { Component } from "react";
import Container from "react-bootstrap/Container";

class ClickableText extends Component {
  constructor(props) {
    super(props);
    this.state = {
      highlight: false
    };
  }
  onPointerOver = () => {
    this.setState({ highlight: true });
  };
  onPointerLeave = () => {
    this.setState({ highlight: false });
  };

  render() {
    const { children, onClick, chosen, active } = this.props;
    const { highlight } = this.state;
    const className = active
      ? highlight
        ? "small-highlight"
        : "small-text"
      : chosen
      ? "green-text"
      : "small-highlight";
    return (
      <Container
        onPointerOver={this.onPointerOver}
        onPointerLeave={this.onPointerLeave}
        onClick={onClick}
        className={className}
      >
        {children}
      </Container>
    );
  }
}

export default ClickableText;
