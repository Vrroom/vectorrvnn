import networkx as nx
from .treeOps import leavesInSubtree, parent, descendants
import copy
import more_itertools, itertools

def hac2nxDiGraph (leaves, links) :
    T = nx.DiGraph()
    T.add_nodes_from(leaves)
    n = max(leaves) + 1
    leaves.extend(list(range(n, 3 * n)))
    for i, link in enumerate(links) : 
        T.add_edge(n + i, leaves[link[0]])
        T.add_edge(n + i, leaves[link[1]])
    for node in T.nodes:  
        T.nodes[node]['pathSet'] = list(leavesInSubtree(T, node))
    return T

def contractGraph (G, nodeSet) : 
    """
    Contract the graph by merging all 
    nodes in the node set.

    We also change the name of the
    contracted node to indicate all
    those original nodes which it had.

    Parameters
    ----------
    G : nx.Graph
        The graph to be contracted.
    nodeSet : list
        Nodes in the graph.
    """
    assert len(nodeSet) > 0, "Node set is empty"
    H = G
    for u, v in zip(nodeSet, nodeSet[1:]) : 
        H = nx.contracted_nodes(H, v, u, self_loops=False)
    mapping = {nodeSet[-1]: tuple(more_itertools.collapse(nodeSet))}
    H = nx.relabel_nodes(H, mapping)
    return H

def subgraph (G, predicate=lambda x : x['weight'] < 1e-3) :
    """ 
    Get a subgraph of G by
    removing edges which don't 
    satisfy a given predicate.

    Useful when filtering those edges
    from the symmetry graph which have
    high error.

    Parameters
    ----------
    G : nx.Graph()
        Normal Graph
    predicate : lambda 
        Predicate over edges to be kept
        in the subgraph
    """
    G_ = copy.deepcopy(G)
    badEdges = list(itertools.filterfalse(lambda e : predicate(G_.edges[e]), G_.edges))
    G_.remove_edges_from(badEdges)
    return G_

def subtreeSize(s, T, subSize) :
    """
    Calculate the size of each
    subtree in the rooted tree T.

    Parameters
    ----------
    s : int
        The current vertex
    T : nx.DiGraph()
        The tree
    subSize : dict
        Store the size of the subtree
        rooted at each vertex.
    """
    subSize[s] = 1
    nbrs = list(T.neighbors(s))
    if T.out_degree(s) != 0 : 
        for u in nbrs :
            subtreeSize(u, T, subSize)
            subSize[s] += subSize[u]

def nxGraph2appGraph (forest) : 
    appGraph = dict(nodes=[], links=[])
    sortedNodes = sorted(list(forest.nodes))
    sortedNodes = list(map(int, sortedNodes))
    for n in sortedNodes :
        nodeType = "path" if forest.out_degree(n) == 0 else "group"
        paths = list(leavesInSubtree(forest, n))
        children = list(forest.neighbors(n))
        paths = list(map(int, paths))
        children = list(map(int, children))
        node = dict(
            id=n, 
            x=0, 
            y=0, 
            type=nodeType, 
            paths=paths, 
            children=children
        )
        appGraph['nodes'].append(node)
    for (u, v) in forest.edges : 
        link = dict(source=int(u), target=int(v), type="group")
        appGraph['links'].append(link)
    for n in sortedNodes : 
        nodeParent = parent(forest, n)
        if nodeParent is not None : 
            appGraph['nodes'][n]['parent'] = int(nodeParent)
    return appGraph

def leqInPO (i, j, po) : 
    """ is i <= j in the partial order """
    return i in descendants(po, j) 

def simplifyPO (po) : 
    """ 
    If i <= j <= k, then remove links like i <= k 
    """
    edgesToBeRemoved = []
    for n in po.nodes : 
        neighbors = list(po.neighbors(n)) 
        for c in neighbors : 
            gc = descendants(po, c) - {c}
            needless = set(neighbors).intersection(gc)
            edgesToBeRemoved.extend([(n, _) for _ in needless])
    po_ = copy.deepcopy(po)
    po_.remove_edges_from(edgesToBeRemoved)
    return po_


