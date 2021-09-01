import React, { Component } from "react";
import { preprocessSVG } from "../utils/svg";
import { createEmptyGraph, setDepths } from "../utils/graph";

function withGraphicFetcher(Wrapped, endpoint) {
  return class extends Component {
    constructor(props) {
      super(props);
      const graphic = preprocessSVG('<svg height="100" width="100"></svg>');
      const forest = createEmptyGraph(graphic);
      this.state = { graphic, forest, id: -1, target: [] };
    }

    fetchSVG = () => {
      fetch(endpoint)
        .then((res) => res.json())
        .then((item) => {
          const { id, svg, forest, target } = item;
          setDepths(forest);
          const graphic = preprocessSVG(svg);
          this.setState({ graphic, forest, id, target });
        });
    };

    componentDidMount() {
      this.fetchSVG();
    }

    render() {
      return <Wrapped fetcher={this.fetchSVG} {...this.state} />;
    }
  };
}

export default withGraphicFetcher;
