import { getWidthHeight, coveringBBox } from "./svg";

function parents(a, graph, edgeFilter) {
  const edges = graph.links.filter(edgeFilter);
  const parentEdges = edges.filter(e => e.target.id === a);
  return parentEdges.map(e => e.source.id);
}

function recomputedPaths(nodes) {
  const helper = a => {
    if (nodes[a].children.length > 0) {
      nodes[a].paths = nodes[a].children.map(helper).flat();
    }
    return nodes[a].paths;
  };
  nodes.forEach ((_, i) => helper(i));
  return nodes;
}

function connected(a, b, graph, edgeFilter, directed) {
  const neighbors = id => {
    const edges = graph.links.filter(edgeFilter);
    let connectedEdges;
    if (directed) {
      connectedEdges = edges.filter(e => e.source.id === id);
    } else {
      connectedEdges = edges.filter(
        e => e.source.id === id || e.target.id === id
      );
    }
    return connectedEdges.map(e =>
      e.source.id === id ? e.target.id : e.source.id
    );
  };
  let marked = Array(graph.nodes.length).map(() => false);
  const dfs = s => {
    marked[s] = true;
    if (s === b) {
      return true;
    }
    return neighbors(s)
      .filter(n => !marked[n])
      .map(dfs)
      .some(x => x);
  };
  return dfs(a);
}

function createEmptyGraph(graphic) {
  const { width, height } = getWidthHeight(graphic.svg.properties);
  const graph = {
    nodes: graphic.paths.map((path, i) => {
      const x = Math.random() * width;
      const y = Math.random() * height;
      const pathWidth = graphic.bboxes[i].width;
      const pathHeight = graphic.bboxes[i].height;
      const ratio = (pathWidth * pathHeight) / (width * height);
      return {
        id: i,
        x,
        y,
        type: "path",
        radius: 3.0 + ratio * (6.0 - 3.0),
        paths: [i],
        children: []
      };
    }),
    links: []
  };
  return graph;
}

function nodeRadius(bboxes, pathIds, svg) {
  const relevantBoxes = pathIds.map(id => bboxes[id]);
  const box = coveringBBox(relevantBoxes);
  const svgBox = getWidthHeight(svg.properties);
  const ratio = (box.height * box.width) / (svgBox.height * svgBox.width);
  return 3.0 + ratio * (6.0 - 3.0);
}

export { recomputedPaths, connected, parents, createEmptyGraph, nodeRadius };
