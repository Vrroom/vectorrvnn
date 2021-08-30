/**
 * @file SVGHandler class implementation.
 *
 * @author Sumit Chaturvedi
 */
import React, { Component } from "react";
import { selectColor, stickColor, boundaryColor } from "../utils/palette";
import addStopPropagation from "../utils/eventModifier";
import { distance, isStylePropertyNotNone, extractViewBox } from "../utils/svg";

function getColor (attr, props) { 
  return isStylePropertyNotNone(attr, props) 
    ? selectColor 
    : "none";
}

/**
 * Responsible for showing the graphic and the paths that are selected.
 * 
 * @extends Component
 */
class SVGHandler extends Component {

  /** 
   * Create cover element for a path if it is selected.
   */
  coverElement = (path) => {
    const child = path.children[0];
    return (
      <g>
        {React.createElement(child.tagName, {
          ...child.properties,
          fill: getColor("fill", child.properties),
          stroke: getColor("stroke", child.properties),
        })}
      </g>
    );
  }

  /**
   * Create React Elements for SVG paths.
   *
   * Set event listeners for various mouse events.
   *
   * @returns {Array}   List of graphic elements as React Elements.
   */
  graphicElements = () => {
    const { paths } = this.props.graphic;
    const { onPointerDown, selected } = this.props;
    return paths.map((path, idx) => {
      const child = path.children[0];
      return (
        <g 
          key={`path-group-${idx}`}
          onPointerDown={addStopPropagation(evt => onPointerDown(evt, idx))}
          transform={path.transform}
        >
          {React.createElement(
            path.tagName,
            { id: `path-${idx}` },
            [React.createElement(child.tagName, child.properties)]
          )}
          {selected.includes(idx) ? this.coverElement(path) : null}
        </g>
      );
    });
  };

  render() {
    const { svg } = this.props.graphic;
    const { svgId } = this.props;
    const children = this.graphicElements();
    return React.createElement(
      svg.tagName,
      {
        ...svg.properties,
        id: svgId,
      },
      children
    );
  }
}

export default SVGHandler;
