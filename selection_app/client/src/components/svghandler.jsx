/**
 * @file SVGHandler class implementation.
 *
 * @author Sumit Chaturvedi
 */
import React, { Component } from "react";
import { selectColor } from "../utils/palette";
import addStopPropagation from "../utils/eventModifier";
import { setAttribute } from "../utils/svg";
import { cloneDeep } from "lodash";

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
    const child = cloneDeep(path.children[0]);
    child.properties = setAttribute(child.properties, "fill", selectColor);
    child.properties = setAttribute(child.properties, "stroke", selectColor);
    return (
      <g>
        {React.createElement(child.tagName, {
          ...child.properties,
        })}
      </g>
    );
  };

  previewElement = (path) => {
    const child = cloneDeep(path.children[0]);
    child.properties = setAttribute(child.properties, "fill", selectColor);
    child.properties = setAttribute(child.properties, "stroke", selectColor);
    return (
      <g strokeOpacity="0.7" fillOpacity="0.7">
        {React.createElement(child.tagName, {
          ...child.properties,
        })}
      </g>
    );
  };

  /**
   * Create React Elements for SVG paths.
   *
   * Set event listeners for various mouse events.
   *
   * @returns {Array}   List of graphic elements as React Elements.
   */
  graphicElements = () => {
    const { paths } = this.props.graphic;
    const { onPointerDown, selected, preview } = this.props;
    return paths.map((path, idx) => {
      const child = path.children[0];
      return (
        <g
          key={`path-group-${idx}`}
          onPointerDown={addStopPropagation((evt) => onPointerDown(evt, idx))}
          onClick={addStopPropagation((evt) => {})}
          transform={path.transform}
        >
          {React.createElement(path.tagName, { id: `path-${idx}` }, [
            React.createElement(child.tagName, child.properties),
          ])}
          {preview.includes(idx) ? this.previewElement(path) : null}
          {selected.includes(idx) ? this.coverElement(path) : null}
        </g>
      );
    });
  };

  render() {
    const { svg } = this.props.graphic;
    const { svgId, onPointerUp, onPointerMove, toolRenderer } = this.props;
    const children = this.graphicElements();
    children.push(toolRenderer());
    return React.createElement(
      svg.tagName,
      {
        ...svg.properties,
        id: svgId,
        onPointerUp,
        onPointerMove,
      },
      children
    );
  }
}

export default SVGHandler;
