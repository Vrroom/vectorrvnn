import React, { Component } from "react";
import Chart from "chart.js";
import io from "socket.io-client";
import Toast from "react-bootstrap/Toast";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Container from "react-bootstrap/Container";
import SVGHandler from "./svghandler";
import GraphHandler from "./graphhandler";
import Buttons from "./buttons";
import { intersection } from "underscore";
import Connections from "./connections";
import {
  getWidthHeight,
  preprocessSVG,
  distance,
  convertCoordinates
} from "../utils/svg";
import { compareClusters } from "../utils/compareClusters";
import { hierarchyforce } from "../utils/hierarchyforce";
import { boxforce } from "../utils/boxforce";
import { cloneDeep } from "lodash";
import {
  recomputedPaths,
  connected,
  createEmptyGraph,
  nodeRadius
} from "../utils/graph";
import * as d3 from "d3";

class Controller extends Component {
  constructor(props) {
    super(props);
    const graphic = preprocessSVG('<svg height="100" width="100"></svg>');
    const graph = createEmptyGraph(graphic);
    this.state = {
      graphic,
      graph,
      selected: [],
      hover: [],
      pointId: graph.nodes.length - 1,
      showSaveToast: false,
      showHelpToast: false,
      online: [],
      id: -1
    };
    this.socket = io();

    this.socket.on("update-online", online => {
      this.setState({ online });
    });

    this.socket.on("recieve-data", data => {
      const { graphic, id } = data;
      const graph = createEmptyGraph(graphic);
      debugger;
      this.setState({
        graphic,
        graph,
        selected: [],
        hover: [],
        pointId: graph.nodes.length - 1,
        showSaveToast: false,
        showHelpToast: false,
        id
      });
      this.updateSimulation(graph, graphic);
    });

    this.socket.on("recieve-graph", thatGraph => {
      const { graph } = this.state;
      const { bs, es } = compareClusters(thatGraph, graph);
      const xs = Array.from(Array(bs.length), (_, i) => i + 2);
      const ctx = document.getElementById("compare-chart").getContext("2d");
      var config = {
        type: "line",
        data: {
          labels: xs,
          datasets: [
            {
              label: "Cluster Compare",
              backgroundColor: 'rgba(0, 0, 0, 1)',
              borderColor: 'rgba(0, 0, 0, 1)',
              data: bs,
              fill: false
            }
          ]
        },
        options: {
          scales: {
            xAxes: [
              {
                display: true,
                scaleLabel: {
                  display: true,
                  labelString: "k"
                }
              }
            ],
            yAxes: [
              {
                display: true,
                scaleLabel: {
                  display: true,
                  labelString: "bk"
                }
              }
            ]
          }
        }
      };
      const myChart = new Chart(ctx, config);
    });

    this.socket.on("request-graph", name => {
      this.socket.emit("recieve-graph", {
        username: name,
        graph: this.state.graph
      });
    });

    this.sim = d3.forceSimulation();
    this.updateSimulation(graph, graphic);
  }

  setStateWithNewSVG = (svgString, id) => {
    const graphic = preprocessSVG(svgString);
    const graph = createEmptyGraph(graphic);
    this.setState({
      graphic,
      graph,
      selected: [],
      hover: [],
      pointId: graph.nodes.length - 1,
      showSaveToast: false,
      showHelpToast: false,
      id
    });
    this.updateSimulation(graph, graphic);
  };

  updateSimulation = (graph, graphic) => {
    const { width, height } = getWidthHeight(graphic.svg.properties);
    const copy = { ...graph };
    this.sim
      .alpha(1)
      .restart()
      .nodes(copy.nodes)
      .force(
        "links",
        d3.forceLink(copy.links).distance(link => {
          return link.type === "normal" ? 30 : 10;
        })
      )
      .force(
        "collide",
        d3.forceCollide().radius(node => node.radius)
      )
      .force("hierarchical", hierarchyforce())
      .force("charge", d3.forceManyBody().strength(-10))
      .force(
        "boxforce",
        boxforce(node => node.radius, width, height)
      )
      .force("forceX", d3.forceX(width / 2).strength(0.1))
      .force("forceY", d3.forceY(height / 2).strength(0.1))
      .on("tick", () => {
        this.setState({ graph: copy, pointId: copy.nodes.length - 1 });
      });
  };

  handleClick = id => {
    /**
     * An id represents a node. A node may be a
     * group node or a path node. A node cannot be
     * selected if it's ancestor or descendent is
     * already selected.
     */
    if (this.mouseMoved) {
      this.mouseMoved = false;
      return;
    }
    const selected = [...this.state.selected];
    const graph = { ...this.state.graph };

    const hasAncestor = selected
      .filter(s => s !== id)
      .map(s => connected(s, id, graph, link => link.type === "group", true))
      .some(x => x);

    const hasChild = selected
      .filter(s => s !== id)
      .map(s => connected(id, s, graph, link => link.type === "group", true))
      .some(x => x);

    if (hasChild || hasAncestor) {
      this.setState({
        showHelpToast: true,
        helpMessage:
          "Cannot select a node/path whose descendent or ancestor has been selected."
      });
      return;
    }

    const isSelected = selected.includes(id);
    if (isSelected) {
      selected.splice(selected.indexOf(id), 1);
    } else {
      selected.push(id);
    }
    this.setState({ selected });
  };

  handlePointerOver = id => {
    const graph = { ...this.state.graph };
    const node = graph.nodes[id];
    if (node.type === "path") {
      const hover = [id];
      this.setState({ hover });
    } else {
      const hover = node.children.map(i => graph.nodes[i].paths).flat();
      this.setState({ hover });
    }
  };

  handlePointerLeave = id => {
    this.setState({ hover: [] });
  };

  handleEdgeDblClick = id => {
    const graph = cloneDeep(this.state.graph);
    const { bboxes, svg } = this.state.graphic;
    let { nodes, links } = graph;
    const groupId = links[id].source.id;

    let nodesToBeDeleted = [];
    let nodeId = groupId;
    while (nodeId) {
      if (nodes[nodeId].children.length <= 2) {
        nodesToBeDeleted.push(nodeId);
      } else {
        break;
      }
      nodeId = nodes[nodeId].parent;
    }

    nodes[groupId].children.splice(
      nodes[groupId].children.indexOf(links[id].target.id),
      1
    );

    let linksToBeDeleted = links
      .filter(
        l =>
          nodesToBeDeleted.includes(l.source.id) ||
          nodesToBeDeleted.includes(l.target.id)
      )
      .map(l => links.indexOf(l));
    linksToBeDeleted.push(id);

    nodes = nodes.filter((_, i) => !nodesToBeDeleted.includes(i));
    links = links.filter((_, i) => !linksToBeDeleted.includes(i));

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
      nodes[i].parent = idMap[nodes[i].parent];
    }

    for (let i = 0; i < links.length; i++) {
      links[i].index = i;
    }

    nodes = recomputedPaths(nodes);

    for (let i = 0; i < nodes.length; i++) {
      nodes[i].radius = nodeRadius(bboxes, nodes[i].paths, svg);
    }

    graph.nodes = nodes;
    graph.links = links;

    this.setState({ selected: [] });
    this.updateSimulation(graph, this.state.graphic);
  };

  fetchFromDB = () => {
    // Fetch random SVG from server.
    fetch("/db")
      .then(res => res.json())
      .then(item => {
        const svg = item.svg;
        const id = item.id;
        this.setStateWithNewSVG(svg, id);
      });
  };

  componentDidMount() {
    this.fetchFromDB();
  }

  handleRandomSVGClick = () => {
    this.fetchFromDB();
  };

  handleGroupClick = () => {
    /**
     * All groups should be distinct and shouldn't
     * share children with some other pre-existing group
     * that isn't their descendent.
     */
    const selected = [...this.state.selected];
    const graph = { ...this.state.graph };
    const graphic = { ...this.state.graphic };
    const nNodes = graph.nodes.length;
    const paths = selected.map(id => graph.nodes[id].paths).flat();

    const othersChildren = graph.nodes
      .filter(n => n.type === "group")
      .map(n => n.children)
      .flat();

    const correctNumber = selected.length >= 2;
    const intersects = intersection(othersChildren, selected).length > 0;

    if (!correctNumber || intersects) {
      if (!correctNumber) {
        this.setState({
          showHelpToast: true,
          helpMessage: "Select atleast two paths/nodes to group."
        });
      } else {
        this.setState({
          showHelpToast: true,
          helpMessage: "The group nodes should form a tree structure."
        });
      }
      return;
    }

    graph.nodes.push({
      id: nNodes,
      x: 0,
      y: 0,
      type: "group",
      radius: nodeRadius(graphic.bboxes, paths, graphic.svg),
      paths,
      children: selected
    });

    selected.forEach(id => {
      graph.links.push({ source: nNodes, target: id, type: "group" });
      graph.nodes[id].parent = nNodes;
    });
    this.updateSimulation(graph, graphic);
  };

  handleClearClick = () => {
    /**
     * Can click this whenever. This function removes
     * all the bounding boxes and the selected nodes.
     */
    const selected = [];
    this.setState({ selected });
  };

  handleSaveClick = () => {
    const data = { id: this.state.id, graph: this.state.graph };
    fetch("/save", {
      method: "post",
      headers: {
        Accept: "application/json, text/plain, */*",
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    })
      .then(res => res.json())
      .then(res => {
        this.setState({ showSaveToast: true });
      });
  };

  onSaveToastClose = () => {
    this.setState({ showSaveToast: false });
  };

  onHelpToastClose = () => {
    this.setState({ showHelpToast: false });
  };

  handlePointerDown = evt => {
    /**
     * We want pointer move events on the SVG
     * to be noted only after the pointer is pressed
     * on a graph node.
     *
     * Hence we have a state variable pointerDown
     * when this happens.
     *
     * We find the graph node which was clicked,
     * if there is such a node, set pointerDown to true
     * and set the pointId state so we can raise this node over
     * others.
     */
    const point = convertCoordinates(
      "svg-graph-element",
      evt.clientX,
      evt.clientY
    );
    const { graph } = this.state;
    for (let i = 0; i < graph.nodes.length; i++) {
      const { x, y } = graph.nodes[i];
      if (distance(point, { x, y }) < graph.nodes[i].radius) {
        this.pointerDown = true;
        this.pointId = i;
        break;
      }
    }
  };

  handlePointerMove = evt => {
    /**
     * The sequence of event handling in the browser is
     * pointerDown -> pointerMove -> pointerUp -> click.
     *
     * We want to find out whether a click happened
     * or a drag happened. For that we have a flag
     * called pointerMoved.
     *
     * This flag is set if it was previously unset and
     * the pointer has moved to a point outside the node's
     * radius.
     */
    if (this.pointerDown) {
      const graph = cloneDeep(this.state.graph);
      const position = convertCoordinates(
        "svg-graph-element",
        evt.clientX,
        evt.clientY
      );
      if (!this.pointerMoved) {
        this.pointerMoved =
          distance(graph.nodes[this.pointId], position) >
          graph.nodes[this.pointId].radius;
      }
      if (this.pointerMoved) {
        graph.nodes[this.pointId].x = position.x;
        graph.nodes[this.pointId].y = position.y;
        this.setState({ graph, pointId: this.pointId });
        this.sim.stop();
      }
    }
  };

  handlePointerUp = evt => {
    /**
     * If pointer was moved, check if the
     * current pointId was dropped on a group and
     * then append it.
     */
    this.pointerDown = false;
    if (this.pointerMoved) {
      const graph = cloneDeep(this.state.graph);
      const thisNode = graph.nodes[this.pointId];
      let thatId;
      let groupFound = false;

      /**
       * Check whether the current pointId has moved close
       * to some group node. This is done by checking whether the
       * circles of the group node and the current pointId
       * intersect.
       */
      for (thatId = 0; thatId < graph.nodes.length; thatId++) {
        const thatNode = graph.nodes[thatId];
        if (
          thatId !== this.pointId &&
          thatNode.type === "group" &&
          distance(
            { x: thatNode.x, y: thatNode.y },
            { x: thisNode.x, y: thisNode.y }
          ) <
            thisNode.radius + thatNode.radius
        ) {
          groupFound = true;
          break;
        }
      }
      /**
       * If such a group is found and the current pointId
       * is not already part of some group, then we
       * check if it can be added to the group and then do so.
       */
      if (groupFound && !thisNode.parent) {
        const thisPaths = graph.nodes[this.pointId].paths;
        const thosePaths = graph.nodes[thatId].paths;
        if (intersection(thisPaths, thosePaths).length === 0) {
          graph.nodes[this.pointId].parent = thatId;
          graph.nodes[thatId].children.push(this.pointId);
          graph.nodes[thatId].paths = graph.nodes[thatId].paths.concat(
            graph.nodes[this.pointId].paths
          );
          graph.links.push({
            source: thatId,
            target: this.pointId,
            type: "group"
          });
        }
      }
      this.updateSimulation(graph, this.state.graphic);
    }
  };

  handleSend = username => {
    const { graphic, id } = this.state;
    this.socket.emit("send-data", { username, graphic, id });
  };

  handleCompare = username => {
    this.socket.emit("request-graph", username);
  };

  render() {
    return (
      <>
        <Container>
          <Row>
            <Col>
              <SVGHandler
                graphic={this.state.graphic}
                graph={this.state.graph}
                selected={this.state.selected}
                hover={this.state.hover}
                onClick={this.handleClick}
                onPointerOver={this.handlePointerOver}
                onPointerLeave={this.handlePointerLeave}
              />
            </Col>
            <Col>
              <GraphHandler
                graphic={this.state.graphic}
                graph={this.state.graph}
                selected={this.state.selected}
                pointId={this.state.pointId}
                onClick={this.handleClick}
                onPointerOver={this.handlePointerOver}
                onPointerLeave={this.handlePointerLeave}
                onEdgeDblClick={this.handleEdgeDblClick}
                onPointerDown={this.handlePointerDown}
                onPointerMove={this.handlePointerMove}
                onPointerUp={this.handlePointerUp}
              />
            </Col>
            <Col>
              <Connections
                handleSend={this.handleSend}
                handleCompare={this.handleCompare}
                online={this.state.online}
                socket={this.socket}
              />
            </Col>
          </Row>
          <Buttons
            clickRandomSVG={this.handleRandomSVGClick}
            clickGroup={this.handleGroupClick}
            clickClear={this.handleClearClick}
            clickSave={this.handleSaveClick}
          ></Buttons>
          <Row>
            <canvas id="compare-chart" width="100" height="20"></canvas>
          </Row>
        </Container>
        <div
          style={{
            position: "absolute",
            top: 0,
            right: 0
          }}
        >
          <Toast
            onClose={this.onSaveToastClose}
            show={this.state.showSaveToast}
            delay={1000}
            autohide
          >
            <Toast.Header>
              <strong className="mr-auto">SAVED IT!</strong>
            </Toast.Header>
          </Toast>
        </div>
        <div
          style={{
            position: "absolute",
            bottom: 0,
            right: 0
          }}
        >
          <Toast
            onClose={this.onHelpToastClose}
            show={this.state.showHelpToast}
            delay={3000}
            autohide
          >
            <Toast.Header>
              <strong className="mr-auto">Help</strong>
            </Toast.Header>
            <Toast.Body>{this.state.helpMessage}</Toast.Body>
          </Toast>
        </div>
      </>
    );
  }
}

export default Controller;
