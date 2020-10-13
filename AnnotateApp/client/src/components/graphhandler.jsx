import React, { Component } from "react";
import { selectColor, edgeColor } from "../utils/palette";
import { coveringBBox, boxCenter } from "../utils/svg";

class GraphHandler extends Component {
  subsetSVG = (pathIds, nodeId) => {
    const { paths } = this.props.graphic;
    pathIds.sort();
    const pathSubset = pathIds.map(i => paths[i]);
    return pathSubset.map((ps, psId) =>
      React.createElement(ps.tagName, {
        ...ps.properties,
        key: `path-${nodeId}-${psId}`
      })
    );
  };

  node2Group = node => {
    const { graphic, selected } = this.props;
    const { onClick, onPointerOver, onPointerLeave } = this.props;
    const { bboxes } = graphic;
    const box = coveringBBox(node.paths.map(i => bboxes[i]));
    const scale = Math.min(1, node.radius / Math.max(box.width, box.height));
    const stroke = selected.includes(node.id) ? selectColor : "none";
    let { cx, cy } = boxCenter(box);
    cx *= scale;
    cy *= scale;
    const tx = node.x - cx;
    const ty = node.y - cy;
    return (
      <g
        key={`vertex-${node.id}`}
        onClick={() => onClick(node.id)}
        onPointerOver={() => onPointerOver(node.id)}
        onPointerLeave={() => onPointerLeave(node.id)}
      >
        <circle
          key={node.id}
          cx={node.x}
          cy={node.y}
          r={node.radius}
          fill="white"
          stroke={stroke}
        />
        <g
          key={`path-group-${node.id}`}
          transform={`translate(${tx} ${ty}) scale(${scale})`}
        >
          {this.subsetSVG(node.paths, node.id)}
        </g>
      </g>
    );
  };

  getVertices = () => {
    const { graph, pointId } = this.props;
    if (graph.nodes.length === 0) {
      return <path />
    }
    const nodes = graph.nodes.filter((_, i) => i !== pointId);
    nodes.push(graph.nodes[pointId]);
    const groups = nodes.map(this.node2Group);
    return groups;
  };

  getEdges = () => {
    const { links } = this.props.graph;
    const { onEdgeDblClick } = this.props;
    return links.map((link, idx) => (
      <line
        key={`link-${idx}`}
        x1={link.source.x}
        y1={link.source.y}
        x2={link.target.x}
        y2={link.target.y}
        stroke={edgeColor}
        onDoubleClick={() => onEdgeDblClick(idx)}
      />
    ));
  };

  render() {
    const { onPointerDown, onPointerMove, onPointerUp } = this.props;
    const { svg } = this.props.graphic;
    return (
      <svg
        width={svg.properties.width}
        height={svg.properties.height}
        viewBox={svg.properties.viewBox}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        id="svg-graph-element"
      >
        {this.getEdges()}
        {this.getVertices()}
      </svg>
    );
  }
}

export default GraphHandler;
