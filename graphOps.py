import networkx as nx
import copy
import more_itertools

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
        H = nx.contracted_nodes(H, v, u)
    mapping = {nodeSet[-1]: list(more_itertools.flatten(nodeSet))}
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

