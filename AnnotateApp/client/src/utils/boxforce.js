function boxforce(radius, width, height) {
  var nodes;

  function force(alpha) {
    for (var i = 0, n = nodes.length; i < n; ++i) {
      var node = nodes[i];
      var r = radius(node);
      node.x = Math.max(r, Math.min(width - r, node.x));
      node.y = Math.max(r, Math.min(height - r, node.y));
    }
  }

  force.initialize = function(_) {
    nodes = _;
  };

  return force;
}

export { boxforce };
