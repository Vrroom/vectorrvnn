import networkx as nx
from copy import deepcopy

def graphCluster (G, algo, doc) :
    """
    Hierarchical clustering of a graph into a dendogram. Given an algorithm to 
    partition a graph into two sets, the this class produces a tree by recursively 
    partitioning the graphs induced on these subsets till each subset contains only a
    single node.

    Parameters
    ----------
    G : nx.Graph
        The path relationship graph
    algo : lambda
        The partitioning algorithm to be used
    doc : svg.Document
        Used to set node attributes.
    """

    def cluster (lst) :
        """
        Recursively cluster the subgraph induced by the vertices present in the list.
        """
        nonlocal idx
        tree.add_node(idx)
        curId = idx
        idx += 1
        if len(lst) != 1 :
            subgraph = G.subgraph(lst)
            l, r = algo(subgraph)
            lId = cluster(list(l))
            rId = cluster(list(r))
            tree.add_edge(curId, lId)
            tree.add_edge(curId, rId)
        tree.nodes[curId]['pathSet'] = deepcopy(lst)
        return curId

    paths = doc.paths()
    tree = nx.DiGraph()
    idx, root = 0, 0
    cluster(list(G.nodes))
    return tree, root
