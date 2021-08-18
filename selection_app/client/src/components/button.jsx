import React, { Component } from "react";
import Tooltip from "react-bootstrap/Tooltip";
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

  onClick = evt => {
    this.props.onClick(evt);
  };

  render() {
    const { active, src, alt } = this.props;
    const hover = this.state.hover;
    const opacity = hover || !active ? 0.3 : 1;
    const width =
      typeof this.props.width === "undefined" ? "50" : this.props.width;
    const height =
      typeof this.props.height === "undefined" ? "50" : this.props.height;
    return (
      <OverlayTrigger overlay={this.tooltip()} placement="bottom">
        <span className="d-inline-block">
          <img
            src={src}
            width={width}
            height={height}
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
