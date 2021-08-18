/**
 * @file Custom force which aims to move parent nodes 
 * above child nodes. 
 *
 * @author Sumit Chaturvedi
 */

/**
 * Force function for d3-force. This function increases
 * the upward velocity of the parent nodes at the expense
 * of child nodes.
 *
 * The graph looks like a rooted tree.
 *
 * @param   {Number}  dy - distance to be maintained 
 * between child and parent.
 *
 * @return  {function}  Hierarchy maintaining function.
 */
function hierarchyforce (dy) {
  var nodes;

  function force(alpha) {
    for (var i = 0, n = nodes.length; i < n; ++i) {
      var p = nodes[i];
      if (p.children) {
        for (var j = 0; j < p.children.length; ++j) {
          var c = p.children[j];
          nodes[c].x += (p.x - nodes[c].x) * alpha;
          nodes[c].y += (p.y - nodes[c].y + dy) * alpha;
        }
      }
    }
  }

  force.initialize = function (_) {
    nodes = _;
  }

  return force;
}

export { hierarchyforce };
