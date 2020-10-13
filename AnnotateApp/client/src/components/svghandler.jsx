import React, { Component } from "react";
import { selectColor } from "../utils/palette";

class SVGHandler extends Component {
  graphicElements = () => {
    const { paths } = this.props.graphic;
    const { onClick, onPointerOver, onPointerLeave, hover } = this.props;
    return paths.map((path, key) => {
      const { fillOpacity, strokeOpacity } = path.properties;
      const hasHover = key => hover.length > 0 && !hover.includes(key);
      return React.createElement(path.tagName, {
        ...path.properties,
        key,
        fillOpacity: hasHover(key) ? 0.1 : fillOpacity,
        strokeOpacity: hasHover(key) ? 0.1 : strokeOpacity,
        onClick: () => onClick(key),
        onPointerOver: () => onPointerOver(key),
        onPointerLeave: () => onPointerLeave(key)
      });
    });
  };

  boundingBoxGroupElement = () => {
    const { bboxes } = this.props.graphic;
    const { graph, selected } = this.props;
    const { onClick, onPointerOver, onPointerLeave } = this.props;
    const pathIds = selected.map(id => graph.nodes[id].paths).flat();
    const reactBoxes = pathIds.map(id => {
      const bbox = bboxes[id];
      const properties = {
        stroke: selectColor,
        strokeWidth: 2,
        pointerEvents: "stroke",
        onClick: () => onClick(id),
        onPointerOver: () => onPointerOver(id),
        onPointerLeave: () => onPointerLeave(id)
      };
      const key = `bbox-${id}`;
      if (bbox.height > 0 && bbox.width > 0) {
        return (
          <rect
            key={key}
            x={bbox.x}
            y={bbox.y}
            width={bbox.width}
            height={bbox.height}
            fill="transparent"
            {...properties}
          />
        );
      } else {
        const x1 = bbox.x;
        const y1 = bbox.y;
        let x2, y2;
        if (bbox.width === 0) {
          x2 = bbox.x;
          y2 = bbox.y + bbox.height;
        } else {
          x2 = bbox.x + bbox.width;
          y2 = bbox.y;
        }
        return (
          <line key={key} x1={x1} y1={y1} x2={x2} y2={y2} {...properties} />
        );
      }
    });
    return (
      <g key="bbox-group" id="bbox-group">
        {reactBoxes}
      </g>
    );
  };

  render() {
    const { svg } = this.props.graphic;
    const children = this.graphicElements();
    children.push(this.boundingBoxGroupElement());
    return React.createElement(
      svg.tagName,
      { ...svg.properties, id: "svg-element" },
      children
    );
  }
}

export default SVGHandler;
