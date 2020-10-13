import React, { Component } from "react";
import Tooltip from "react-bootstrap/Tooltip";
import Container from "react-bootstrap/Container";
import OverlayTrigger from "react-bootstrap/OverlayTrigger";

class Button extends Component {
  constructor(props) {
    super(props);
    this.state = { hover: false };
  }

  tooltip = () => {
    return <Tooltip>{this.props.name}</Tooltip>;
  };

  onPointerOver = () => {
    this.setState({ hover: true });
  };

  onPointerLeave = () => {
    this.setState({ hover: false });
  };

  onClick = () => {
    this.setState({ show: true });
    this.props.onClick();
  };

  render() {
    const { active, onClick, src, alt } = this.props;
    const hover = this.state.hover;
    const opacity = hover || !active ? 0.5 : 1;
    return (
      <OverlayTrigger overlay={this.tooltip()} placement="bottom">
        <span className="d-inline-block">
          <img
            src={src}
            width="50"
            height="50"
            alt={alt}
            onClick={active ? this.onClick : null}
            onPointerOver={this.onPointerOver}
            onPointerLeave={this.onPointerLeave}
            style={{ opacity }}
          />
        </span>
      </OverlayTrigger>
    );
  }
}

export default Button;
