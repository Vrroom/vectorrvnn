/**
 * @file SVGHandler class implementation.
 *
 * @author Sumit Chaturvedi
 */
import React, { Component } from "react";
import { selectColor, stickColor, boundaryColor } from "../utils/palette";
import addStopPropagation from "../utils/eventModifier";
import { distance, isStylePropertyNotNone, extractViewBox } from "../utils/svg";

/**
 * This component handles the SVG graphic that the users have
 * to annotate.
 *
 * Whenever the user selects a path by clicking on it,
 * a bounding box is added to the UI, marking the selected
 * path.
 *
 * Likewise, whenever the user hovers over a path, the
 * opacity of those paths is reduced to give them some
 * feedback.
 *
 * What the SVG paths look like and whether to display their
 * bounding boxes are handled here.
 *
 * @extends Component
 */
class SVGHandler extends Component {
  /**
   * Create React Elements for SVG paths.
   *
   * Set the fillOpacity and strokeOpacity of paths
   * depending on whether the user is hovering over them.
   *
   * Set event listeners for various mouse events.
   *
   * @returns {Array}   List of graphic elements as React Elements.
   */
  graphicElements = () => {
    const { paths } = this.props.graphic;
    const { onPointerDown, selected, forest } = this.props;
    const pathIdx = selected.map(id => forest.nodes[id].paths).flat();
    return paths.map((path, key) => {
      const child = path.children[0];
      const coverElement = (
        <g transform={path.transform}>
          {React.createElement(child.tagName, {
            ...child.properties,
            fill: isStylePropertyNotNone("fill", child.properties)
              ? selectColor
              : "none",
            stroke: isStylePropertyNotNone("stroke", child.properties)
              ? selectColor
              : "none",
            key: `cover-${key}-element`,
            id: `cover-${key}-element`,
            onPointerDown: addStopPropagation(evt => onPointerDown(evt, key))
          })}
        </g>
      );
      return (
        <g key={`path-group-${key}`}>
          {React.createElement(
            path.tagName,
            {
              transform: path.transform,
              key,
              id: `path-${key}`,
              onPointerDown: addStopPropagation(evt => onPointerDown(evt, key))
            },
            [React.createElement(child.tagName, child.properties)]
          )}
          {pathIdx.includes(key) ? coverElement : null}
        </g>
      );
    });
  };

  scribbleElements = () => {
    const { scribbleStrokes } = this.props;
    const strokeElements = scribbleStrokes.map((stroke, strokeID) => {
      return (
        <g key={`stroke-${strokeID}`}>
          {stroke.map((circle, id) => (
            <circle
              key={`circles-${id}`}
              cx={circle.x}
              cy={circle.y}
              r="2%"
              fill="blue"
              fillOpacity="0.1"
            />
          ))}
        </g>
      );
    });
    return (
      <g key="scribble-circles-group" id="scribble-circles-group">
        {strokeElements}
      </g>
    );
  };

  slider = () => {
    const { cx, cy, p, showSlider } = this.props;
    if (!showSlider) {
      return <g id="slider" />;
    }
    const diffX = p.x - cx;
    const diffY = p.y - cy;
    const p_ = { x: cx - diffX, y: cy - diffY };
    const r = distance(p, p_) / 2 + 1e-2;
    const perp = { x: (-2 * diffY) / r, y: (2 * diffX) / r };
    return (
      <g id="slider" stroke={stickColor} strokeWidth="2%" fill="transparent">
        <line x1={p.x} y1={p.y} x2={p_.x} y2={p_.y} />
        <line
          x1={p.x - perp.x}
          y1={p.y - perp.y}
          x2={p.x + perp.x}
          y2={p.y + perp.y}
        />
        <line
          x1={p_.x - perp.x}
          y1={p_.y - perp.y}
          x2={p_.x + perp.x}
          y2={p_.y + perp.y}
        />
      </g>
    );
  };

  boundaryBox = () => {
    const { svg } = this.props.graphic;
    const vb = extractViewBox(svg.properties.viewBox);
    const linesegs = [
      { x1: vb[0], y1: vb[1], x2: vb[0] + 5, y2: vb[1] },
      { x1: vb[0], y1: vb[1], x2: vb[0], y2: vb[1] + 5 },
      { x1: vb[0], y1: vb[1] + 100, x2: vb[0] + 5, y2: vb[1] + 100 },
      { x1: vb[0], y1: vb[1] + 100, x2: vb[0], y2: vb[1] + 95 },
      { x1: vb[0] + 100, y1: vb[1], x2: vb[0] + 95, y2: vb[1] },
      { x1: vb[0] + 100, y1: vb[1], x2: vb[0] + 100, y2: vb[1] + 5 },
      { x1: vb[0] + 100, y1: vb[1] + 100, x2: vb[0] + 100, y2: vb[1] + 95 },
      { x1: vb[0] + 100, y1: vb[1] + 100, x2: vb[0] + 95, y2: vb[1] + 100 }
    ];
    return (
      <g>
        {linesegs.map((seg, i) => (
          <line
            key={`line-${i}`}
            {...seg}
            stroke={boundaryColor}
            strokeLinecap="round"
            strokeWidth="3%"
          />
        ))}
      </g>
    );
  };

  /**
   * Render the SVG.
   *
   * @return {Component}
   */
  render() {
    const { svg } = this.props.graphic;
    const { svgId } = this.props;
    const { onSVGPointerDown, onPointerMove, onPointerUp } = this.props;
    const children = this.graphicElements();
    children.push(this.boundaryBox());
    children.push(this.slider());
    children.push(this.scribbleElements());
    return React.createElement(
      svg.tagName,
      {
        ...svg.properties,
        id: typeof svgId === "undefined" ? "svg-element" : svgId,
        onPointerMove,
        onPointerUp,
        onPointerDown: addStopPropagation(onSVGPointerDown)
      },
      children
    );
  }
}

export default SVGHandler;
