import React, { Component } from "react";
import Modal from "react-bootstrap/Modal";
import Table from "react-bootstrap/Table";
import clear from "../icons/clear.svg";
import group from "../icons/group.svg";
import random from "../icons/random.svg";
import related from "../icons/related.svg";
import save from "../icons/save.svg";

class Help extends Component {
  render() {
    const { onHide, show } = this.props;
    return (
      <Modal
        size="lg"
        show={show}
        onHide={onHide}
        aria-labelledby="example-modal-sizes-title-lg"
      >
        <Modal.Header closeButton>
          <Modal.Title id="example-modal-sizes-title-lg">Help</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <p>
            We are trying to collect data on perceptual grouping in{" "}
            <a href="https://en.wikipedia.org/wiki/Vector_graphics">
              Vector Graphics
            </a>
            . Vector Graphics are composed of splines/paths. We want to know
            which paths go together. That is, which paths group to find a{" "}
            <b>meaningful</b> perceptual entity. Of course, the definition of{" "}
            <b>meaningful</b> is tricky. This makes the problem of
            algorithmically grouping things challenging.
          </p>

          <p>
            This app helps you connect related paths and visualize the result.
            The original image is on the left. Each black bubble on the right
            represents an individual path in the original image. You can select
            a path either by clicking on the image or on the bubbles. Once you
            select paths, you can either <b>group</b> them or add a{" "}
            <b>relation</b> between them.
          </p>
          <Table variant="dark">
            <tbody>
              <tr>
                <td>
                  <img src={random} height="50" width="50" alt="Random" />
                </td>
                <td>Pull out another random image from the database.</td>
              </tr>
              <tr>
                <td>
                  <img src={group} height="50" width="50" alt="Group"/>
                </td>
                <td>
                  Group selected paths. You have to select <b>two or more</b>{" "}
                  paths to group. A new bubble is inserted. This bubble
                  represents all the paths in the group.
                </td>
              </tr>
              <tr>
                <td>
                  <img src={clear} height="50" width="50" alt="Clear"/>
                </td>
                <td>Clear the current selection.</td>
              </tr>
              <tr>
                <td>
                  <img src={save} height="50" width="50" alt="Save"/>
                </td>
                <td>Save the relationship graph created.</td>
              </tr>
            </tbody>
          </Table>
        </Modal.Body>
      </Modal>
    );
  }
}

export default Help;
