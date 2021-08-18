import React, { Component } from "react";
import Toast from "react-bootstrap/Toast";

class SaveToast extends Component {
  render() {
    return (
      <div
        style={{
          position: "absolute",
          top: 0,
          right: 0
        }}
      >
        <Toast
          onClose={this.props.onClose}
          show={this.props.show}
          delay={2000}
          autohide={true}
        >
          <Toast.Header>
            <strong className="mr-auto">Thank You!</strong>
          </Toast.Header>
        </Toast>
      </div>
    );
  }
}

export default SaveToast;
