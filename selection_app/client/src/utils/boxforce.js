/**
 * @file Custom box confining force function for d3-force.
 *
 * @author Sumit Chaturvedi
 */

/**
 * Force function for d3-force. This force ensures
 * that the balls of a certain radius stay within
 * a box of given dimensions.
 *
 * @example
 * d3.forceSimulation()
 *   .force("boxforce", boxforce(n => n.radius, 100, 100))
 *
 * @param   {function}  radius - Specify radius for a node.
 * @param   {Number}    width - Width of the box.
 * @param   {Number}    height - Height of the box.
 *
 * @returns {function}  Box confining function.
 */
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
