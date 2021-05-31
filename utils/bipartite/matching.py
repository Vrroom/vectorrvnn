"""
Earlier in my experiments with recursive
networks, I was using this to line up 
the features of encoded and decoded nodes
using L2 Norm.
"""
import networkx as nx
from networkx.algorithms import bipartite

def optimalBipartiteMatching (costTable) :
    """
    Return the minimum cost bipartite 
    matching.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the 
        cost of matching that pair
        together.
    """
    G = nx.Graph()
    for key, val in costTable.items() :
        i, j = key
        G.add_node(i, bipartite=0)
        G.add_node(str(j), bipartite=1)
        G.add_edge(i, str(j), weight=val)
    matchingWithDups = bipartite.minimum_weight_full_matching(G)
    matching = dict()
    for key, val in matchingWithDups.items() : 
        if not isinstance(key, str):
            matching[key] = int(val)
    return matching

def bestAssignmentCost (costTable) :
    """
    Compute the minimum total
    cost assignment.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the 
        cost of matching that pair
        together.
    """
    matching = optimalBipartiteMatching(costTable)
    cost = sum([costTable[e] for e in matching.items()])
    return cost

