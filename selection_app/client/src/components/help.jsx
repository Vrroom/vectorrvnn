import React, { Component } from "react";
import Modal from "react-bootstrap/Modal";
import ResponsiveEmbed from "react-bootstrap/ResponsiveEmbed";
import example from "../videos/example.mp4";

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
            The image on the left has been decomposed into parts. Each bubble on
            the right contains one part from the image. Your task is to
            reconstruct the image by combining the bubbles. Parts can be
            combined by:
            <ol>
              <li>
                Selecting them either by clicking on them in the image or by
                clicking on the bubble containing that part.
              </li>
              <li> Clicking on the Group button </li>
            </ol>
          </p>
          <p>
            The easiest way to do this is to select all parts and group them all
            at once. However, this is not what we are looking for. We want that
            each group that you form have a few parts. Ideally select 2-5 parts
            to group. You should be able to rationalize why you created a group.
            For example:
            <ul>
              <li> Subparts in the group had the same color. </li>
              <li> They served the same functon. </li>
              <li> They were closeby. </li>
              <li> They formed a symmetry. </li>
            </ul>
            The list is endless. If you are able to describe, in words, why you
            are grouping a set of parts, then you should go ahead and group
            them!
          </p>
          <p>
            In case you made a mistake and grouped a set of parts that you
            didn't mean to, you can break the group by double clicking on the
            corresponding bubble.
          </p>
          <p>
            After you are done, press the Save button. Press the Shuffle button
            to get a new image.
          </p>
          <h1> Example </h1>
          <ResponsiveEmbed aspectRatio="16by9">
            <embed type="video/mp4" src={example} />
          </ResponsiveEmbed>
        </Modal.Body>
      </Modal>
    );
  }
}

export default Help;
