// TODO Change IP address
import React, { Component } from "react";
import Row from "react-bootstrap/Row";
import Container from "react-bootstrap/Container";
import heart from "../icons/heart.svg";
import Button from "./button";
import {
  FacebookShareButton,
  FacebookIcon,
  TwitterIcon,
  WhatsappIcon,
  TwitterShareButton,
  WhatsappShareButton,
} from "react-share";

class ByeSlide extends Component {
  render() {
    return (
      <Container className="normal-text">
        <Row className="justify-content-center">
          <Button 
            src={heart}
            name="Purple Heart"
            active={true}
            alt="Purplet Heart"
            onClick={() => {}}
            width="100"
            height="100"
          />
        </Row>
        <Row className="justify-content-around">
          <FacebookShareButton
            url="http://34.72.137.149/"
            quote="SVG Selection Study"
            className="Demo__some-network__share-button"
          >
            <FacebookIcon size={32} round />
          </FacebookShareButton>
          <div className="Demo__some-network">
            <TwitterShareButton
              url="http://34.72.137.149/"
              title="SVG Selection Study"
              className="Demo__some-network__share-button"
            >
              <TwitterIcon size={32} round />
            </TwitterShareButton>

            <div className="Demo__some-network__share-count">&nbsp;</div>
          </div>
          <div className="Demo__some-network">
            <WhatsappShareButton
              url="http://34.72.137.149/"
              title="SVG Selection Study"
              separator=":: "
              className="Demo__some-network__share-button"
            >
              <WhatsappIcon size={32} round />
            </WhatsappShareButton>
            <div className="Demo__some-network__share-count">&nbsp;</div>
          </div>
        </Row>
      </Container>
    );
  }
}

export default ByeSlide;
