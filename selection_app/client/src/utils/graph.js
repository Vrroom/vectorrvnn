/*
 * @file Functions for manipulating the graph over SVG paths.
 *
 * @author Sumit Chaturvedi
 */
import { getWidthHeight, coveringBBox, distance } from "./svg";
import { disjoint, isSubset } from "./listOps";
import { cloneDeep, range } from "lodash";
import { nodeColors } from "./palette";

function descendants(graph, id) {
  const { nodes } = graph;
  const helper = a => {
    if (nodes[a].children.length > 0) {
      const list = nodes[a].children.map(helper).flat();
      list.push(a);
      return list;
    }
    return nodes[a].paths;
  };
  const all = helper(id);
  return all;
}

function lca(forest, ids) {
  return forest.nodes
    .map(node => {
      return { self: node, result: descendants(forest, node.id) };
    })
    .filter(thing => isSubset(ids, thing.result))
    .sort((a, b) => a.self.depth - b.self.depth)
    .pop().self.id;
}

function setDepths(forest) {
  for (let i = 0; i < forest.nodes.length; i++) {
    depth(i, forest);
  }
}

function depth(nodeId, forest) {
  let nodeDepth = forest.nodes[nodeId].depth;
  if (typeof nodeDepth === "undefined") {
    const parentId = forest.nodes[nodeId].parent;
    if (!isRoot(nodeId, forest)) {
      nodeDepth = 1 + depth(parentId, forest);
    } else {
      nodeDepth = 0;
    }
    forest.nodes[nodeId].depth = nodeDepth;
  }
  return nodeDepth;
}

/*
 * Calculate paths in each subtree.
 *
 * The nodes form a hierarchy over a set
 * of paths. Hence each node in the hierarchy
 * has a subset of SVG paths associated with them.
 *
 * For example, the root of the tree will contain
 * all SVG paths.
 *
 * For each node in nodes, this subset is calculated
 * and the node's path property is updated with this
 * subset.
 *
 * @param   {Object}  nodes - List of graph nodes.
 *
 * @return  {Object}  A list of graph nodes populated with the
 * correct path subsets.
 */
function gatherSubtreePaths(nodes) {
  const helper = a => {
    if (nodes[a].children.length > 0) {
      nodes[a].paths = nodes[a].children.map(helper).flat();
    }
    return nodes[a].paths;
  };
  nodes.forEach((_, i) => helper(i));
  return nodes;
}

/*
 * Check if two nodes a and b are connected in a graph.
 *
 * @param   {Number}  a - Id for node a.
 * @param   {Number}  b - Id for node b.
 * @param   {Object}  graph - Graph of the form { nodes, links }.
 * @param   {boolean} directed - Whether the graph is directed.
 *
 * @returns {boolean} Whether the nodes are connected or not.
 */
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
  let marked = new Array(graph.nodes.length).fill(false);
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

/*
 * Create an empty graph from a graphic.
 *
 * The empty graph is over the path set of the SVG.
 * Initially, there are no edges.
 *
 * @param   {Object}  graphic - An object containing the SVG document,
 * paths and their bounding boxes.
 *
 * @return  {Object}  Has properties nodes and links. One node per path
 * and an empty list for links.
 */
function createEmptyGraph(graphic) {
  const { width, height } = getWidthHeight(graphic.svg.properties);
  const nPaths = graphic.paths.length;
  const components = range(nPaths);
  const minRadius = 100 / (4 * nPaths);
  let nodes = graphic.paths.map((path, i) => {
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
      fill: "#f3f9fb",
      radius: minRadius + ratio * minRadius,
      paths: [i],
      children: [],
      visible: 1
    };
  });
  let links = [];
  for (let i = 0; i < components.length; i++) {
    const component = components[i];
    if (component.length > 1) {
      const source = nodes.length;
      nodes.push({
        id: source,
        x: 0,
        y: 0,
        type: "group",
        fill: nodeColors.group,
        radius: 0,
        paths: component,
        children: component,
        visible: 1
      });
      for (let j = 0; j < component.length; j++) {
        const target = component[j];
        nodes[target].parent = source;
        links.push({ source, target, type: "group" });
      }
    }
  }
  const graph = { nodes, links };
  return updateVisualProperties(graph, graphic);
}

/*
 * Calculates the radius of a given node.
 *
 * In the GraphHandler component, there are nodes
 * corresponding to subsets of SVG paths.
 *
 * The radius depends on the ratio between the
 * area of the path subset of the node and the
 * area of the entire viewbox. Additionally, it
 * also dependents on the graph.
 *
 * @param   {Object}  node - node whose radius is to be calculated.
 * @param   {Object}  graph - radius is weighted by the depth of the node
 * in the graph. This is to make sure that deeper nodes look smaller.
 * @param   {Object}  graphic - graphic
 *
 * @return  {Number}  Radius for the node with these pathIds.
 */
function nodeRadius(node, nNodes, graphic) {
  const pathIds = node.paths;
  const { bboxes, svg } = graphic;
  const relevantBoxes = pathIds.map(id => bboxes[id]);
  const box = coveringBBox(relevantBoxes);
  const svgBox = getWidthHeight(svg.properties);
  const ratio = (box.height * box.width) / (svgBox.height * svgBox.width);
  const minRadius = Math.max(4.0, 100 / (4 * nNodes));
  const r = minRadius + ratio * minRadius;
  return r;
}

function suggestions2nodes(suggestions, graphic) {
  let nodes = suggestions.map((s, i) => {
    return { paths: s, id: i };
  });
  const nNodes = nodes.length;
  const radii = nodes.map(node => nodeRadius(node, nNodes, graphic));
  nodes = nodes.map((node, i) => {
    return {
      ...node,
      radius: radii[i],
      x: 0,
      y: 0,
      type: "group",
      fill: nodeColors.group,
    };
  });
  return nodes;
}

/*
 * Check whether node can be added as a child of another node.
 *
 * For this to be allowed, the node being added to has to be
 * a group node, the added node must be the root node of its
 * subtree and their paths shouldn't intersect.
 *
 * @param   {Object}  node - query node.
 * @param   {Object}  to - target node.
 *
 * @return  {Boolean}  Whether the merger can happen.
 */
function canAddNodeAsChild({ node, to } = {}) {
  if (to.type === "group" && typeof node.parent === "undefined") {
    const nodePaths = node.paths;
    const toPaths = to.paths;
    return disjoint(nodePaths, toPaths);
  }
  return false;
}

/*
 * Delete link from forest.
 *
 * Disconnecting the two nodes in the forest
 * may have a ripple effect and may lead to
 * clearance of many other links since no
 * tree in the forest is allowed to have
 * less than two children.
 *
 * @param   {Object}  forest - Collection of trees.
 * @param   {Number}  linkId - Link index to be deleted.
 *
 * @returns {Object}  New Forest.
 */
function deleteLinkFromForest(forest, linkId) {
  let graph = cloneDeep(forest);
  let { nodes, links } = graph;
  const groupId = links[linkId].source.id;

  // Deleting a node can lead to a chain reaction
  // where a lot of other nodes are deleted. This
  // is because no group node can have less than
  // two children. I go up the tree, following the
  // parent link and add all those nodes which would
  // be deleted.
  let nodesToBeDeleted = [];
  let nodeId = groupId;
  while (typeof nodeId !== "undefined") {
    if (nodes[nodeId].children.length <= 2) {
      nodesToBeDeleted.push(nodeId);
    } else {
      break;
    }
    nodeId = nodes[nodeId].parent;
  }

  nodes[groupId].children.splice(
    nodes[groupId].children.indexOf(links[linkId].target.id),
    1
  );

  let linksToBeDeleted = links
    .filter(
      l =>
        nodesToBeDeleted.includes(l.source.id) ||
        nodesToBeDeleted.includes(l.target.id)
    )
    .map(l => links.indexOf(l));
  linksToBeDeleted.push(linkId);

  for (let i = 0; i < linksToBeDeleted.length; i++) {
    const link = links[linksToBeDeleted[i]];
    nodes[link.target.id].parent = undefined;
  }

  nodes = nodes.filter((_, i) => !nodesToBeDeleted.includes(i));
  links = links.filter((_, i) => !linksToBeDeleted.includes(i));

  // After deleting the nodes and links, have to
  // reset the ids to be consistent with the array
  // ordering. Also have to ensure that the
  // paths, children, parent fields are updated.
  const idMap = {};
  for (let i = 0; i < nodes.length; i++) {
    idMap[nodes[i].id] = i;
  }

  for (let i = 0; i < nodes.length; i++) {
    nodes[i].id = idMap[nodes[i].id];
    nodes[i].index = nodes[i].id;
    nodes[i].children = nodes[i].children
      .filter(cId => !nodesToBeDeleted.includes(cId))
      .map(cId => idMap[cId]);
    if (typeof nodes[i].parent !== "undefined") {
      nodes[i].parent = idMap[nodes[i].parent];
    }
  }

  for (let i = 0; i < links.length; i++) {
    links[i].index = i;
  }

  nodes = gatherSubtreePaths(nodes);

  graph.nodes = nodes;
  graph.links = links;

  return graph;
}

/*
 * Add node as child of other nearby node.
 *
 * If the given node intersects with a node,
 * then if possible, it should be added as a
 * child node of the other node.
 *
 * Of course, the other node has to be a group
 * node for this to be possible.
 *
 * @param   {Object}  forest - The forest.
 * @param   {Object}  nodeId - Id of node that
 * is potentially going to be added as a child node.
 *
 * @returns {Object}  New forest.
 */
function addNodeAsChildIfNearby(forest, nodeId) {
  let graph = cloneDeep(forest);
  const thisNode = graph.nodes[nodeId];
  const closeNodes = graph.nodes.filter(
    node =>
      distance(thisNode, node) < thisNode.radius + node.radius &&
      node.id !== nodeId
  );
  if (closeNodes.length === 1) {
    const thatNode = closeNodes.pop();
    if (canAddNodeAsChild({ node: thisNode, to: thatNode })) {
      thisNode.parent = thatNode.id;
      thatNode.children.push(nodeId);
      thatNode.paths = thatNode.paths.concat(thisNode.paths);
      graph.links.push({
        source: thatNode.id,
        target: thisNode.id,
        type: "group"
      });
    }
  }
  return graph;
}

/*
 * Add group node with nodes as children if possible.
 *
 * @param   {Object}  forest - The forest.
 * @param   {Array}   nodes - List of nodes to be grouped.
 *
 * @returns {Object}  New forest.
 */
function groupNodes(forest, nodes) {
  const graph = cloneDeep(forest);
  const nNodes = graph.nodes.length;
  const paths = nodes.map(id => graph.nodes[id].paths).flat();
  const othersChildren = graph.nodes
    .filter(n => n.type === "group")
    .map(n => n.children)
    .flat();
  const correctNumber = nodes.length >= 2;
  const nodeDisjoint = disjoint(othersChildren, nodes);
  if (correctNumber && nodeDisjoint) {
    graph.nodes.push({
      id: nNodes,
      x: 0,
      y: 0,
      type: "group",
      fill: nodeColors.group,
      radius: 0,
      paths,
      children: nodes,
      visible: 1
    });
    nodes.forEach(id => {
      graph.links.push({ source: nNodes, target: id, type: "group" });
      graph.nodes[id].parent = nNodes;
    });
  }
  return graph;
}

/*
 * Compute fill attribute of graph nodes.
 *
 * Nodes can have two possible fill values
 * depending on whether they are contracted or not.
 *
 * @param   {Object}  graph - graph object.
 *
 * @returns {Object}  Graph with updated node fill.
 */
function computeFill(graph) {
  const { nodes } = graph;
  for (let id = 0; id < nodes.length; id++) {
    const node = nodes[id];
    node.fill = node.contracted ? node.contractedGroup : nodeColors.group;
  }
  return graph;
}

/*
 * Compute radius of graph nodes.
 *
 * @param   {Object}  graph - graph object.
 * @param   {Object}  graphic - graphic object.
 *
 * @returns {Object}  Graph with updated node radii.
 */
function computeNodeRadii(graph, graphic) {
  const { nodes } = graph;
  for (let id = 0; id < nodes.length; id++) {
    const node = nodes[id];
    node.radius = nodeRadius(node, graph, graphic);
  }
  return graph;
}

/**
 * Update visual properties of the graph.
 *
 * Each time the graph is updated, its visual properties
 * such as the radius of the nodes and fills have to
 * be recalculated.
 *
 * @param   {Object}  graph - graph object.
 * @param   {Object}  graphic - graphic object.
 *
 * @returns {Object}  Graph with updated visual properties.
 */
function updateVisualProperties(forest, graphic) {
  let graph = cloneDeep(forest);
  graph = computeFill(graph);
  graph = computeNodeRadii(graph, graphic);
  return graph;
}

/**
 * Determine whether the node is a root of some tree
 *
 * Basically, check whether parent is undefined. If
 * it is return true, else false.
 *
 * @param   {Number}  node - node id.
 * @param   {Object}  graph - graph object.
 *
 * @returns {Boolean}  Whether the node is a root.
 */
function isRoot(node, graph) {
  return typeof graph.nodes[node].parent === "undefined";
}

/**
 * Find the root with this node in its subtree.
 *
 * @param   {Number}  node - node id.
 * @param   {Object}  graph - graph object.
 *
 * @returns {Numeber}  Id of the root.
 */
function findRoot(node, graph) {
  while (!isRoot(node, graph)) {
    node = graph.nodes[node].parent;
  }
  return node;
}

export {
  connected,
  createEmptyGraph,
  nodeRadius,
  descendants,
  canAddNodeAsChild,
  deleteLinkFromForest,
  addNodeAsChildIfNearby,
  groupNodes,
  updateVisualProperties,
  isRoot,
  findRoot,
  depth,
  setDepths,
  lca,
  suggestions2nodes
};
