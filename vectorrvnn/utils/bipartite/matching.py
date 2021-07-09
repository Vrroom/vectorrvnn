"""
Earlier in my experiments with recursive networks,
I was using this to line up the features of encoded 
and decoded nodes using L2 Norm.
"""
import networkx as nx
from uuid import uuid4
from networkx.algorithms import bipartite
from more_itertools import unzip

def optimalBipartiteMatching (costTable) :
    """
    Return the minimum cost bipartite matching.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the cost of matching 
        that pair together.
    """
    G = nx.Graph()
    A, B = list(map(set, unzip(costTable.keys())))
    aMap = dict(zip(A, [str(_) + "-" + str(uuid4()) for _ in A]))
    aInvMap = dict(map(reversed, aMap.items()))
    bMap = dict(zip(B, [str(_) + "-" + str(uuid4()) for _ in B]))
    bInvMap = dict(map(reversed, bMap.items()))
    for key, cost in costTable.items() :
        i, j = key
        G.add_node(aMap[i], bipartite=0)
        G.add_node(bMap[j], bipartite=1)
        G.add_edge(aMap[i], bMap[j], weight=cost)
    matchingWithDups = bipartite.minimum_weight_full_matching(G)
    matching = dict()
    for i in A : 
        if aMap[i] in matchingWithDups : 
            matching[i] = bInvMap[matchingWithDups[aMap[i]]]
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

