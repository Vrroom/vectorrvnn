function hierarchyforce () {
  var nodes;

  function force(alpha) {
    for (var i = 0, n = nodes.length; i < n; ++i) {
      if (nodes[i].children) {
        for (var j = 0; j < nodes[i].children.length; ++j) {
          nodes[nodes[i].children[j]].vy += alpha;
        }
        nodes[i].vy -= alpha;
      }
    }
  }

  force.initialize = function (_) {
    nodes = _;
  }

  return force;
}

export { hierarchyforce };
