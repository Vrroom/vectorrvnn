from pulp import *
from sklearn import metrics
import numpy as np
from copy import deepcopy
import pickle
import pandas as pd
import svgpathtools as svg
import random
from itertools import product
import networkx as nx
import more_itertools
from .treeOps import *
from vectorrvnn.utils.datastructures import *

def ted (t1, t2) : 
    """
    Faster unordered tree edit distance algorithm. 

    Based on https://arxiv.org/pdf/1706.03473.pdf
    """
    def d (x, y) : 
        """ 
        Cost of relabeling node x with node y is
        the symmetric difference of the pathSets divided
        by their union.
        """
        ps1 = set(t1.nodes[x]['pathSet'])
        ps2 = set(t2.nodes[y]['pathSet'])
        return len(ps1 ^ ps2) / len(ps1 | ps2)

    def variableMatrix(row, col) : 
        """
        row and col are the names used to index 
        the dataframe. 
        """
        var = [[LpVariable(f'm_{x}_{y}', cat='Binary') for y in col] for x in row]
        return pd.DataFrame(data=var, columns=col, index=row)
   
    def computeW (x, y) : 
        """
        Compute W matrix recursively and 
        memoize the results.
        """
        nonlocal W
        if np.isinf(W.loc[x, y]) : 
            if t1.out_degree(x) == 0 or t2.out_degree(y) == 0 : 
                W.loc[x, y] = 2 - d(x, y)
            else : 
                set1 = list(descendants(t1, x) - {x})
                set2 = list(descendants(t2, y) - {y})
                v = variableMatrix(set1, set2)
                W_ = np.array([[computeW(x_, y_) for y_ in set2] for x_ in set1])
                prob = LpProblem(f'TreeEditDistance_Sub_Problem_{x}_{y}', LpMaximize)
                # Subproblem objective
                prob += (W_ * v).values.sum() + (2 - d(x, y))
                # Subproblem constaints
                for l in leavesInSubtree(t1, x) : 
                    P = list(set(pathInTree(t1, x, l)) - {x})
                    prob += v.loc[P, set2].values.sum() <= 1
                for l in leavesInSubtree(t2, y) : 
                    P = list(set(pathInTree(t2, y, l)) - {y})
                    prob += v.loc[set1, P].values.sum() <= 1
                prob.solve(PULP_CBC_CMD(msg=False))
                W.loc[x, y] = prob.objective.valueOrDefault()
        return W.loc[x, y]
    
    n, m = t1.number_of_nodes(), t2.number_of_nodes()
    leaves1, leaves2 = leaves(t1), leaves(t2)
    r1, r2 = findRoot(t1), findRoot(t2)
    W = pd.DataFrame(data=-np.inf * np.ones((n, m)), columns=t2.nodes, index=t1.nodes)
    for x, y in itertools.product(t1.nodes, t2.nodes) :
        computeW(x, y)
    # Use the W matrix to calculate the edit distance
    v = variableMatrix(t1.nodes, t2.nodes)
    prob = LpProblem("TreeEditDistance", LpMaximize)
    # Add the objectives
    prob += (W * v).values.sum()
    # Add the constraints
    for l in leaves1 : 
        P = pathInTree(t1, r1, l)
        prob += v.loc[P, list(t2.nodes)].values.sum() <= 1
    for l in leaves2 : 
        P = pathInTree(t2, r2, l)
        prob += v.loc[list(t1.nodes), P].values.sum() <= 1
    prob.solve(PULP_CBC_CMD(msg=False))
    opt = prob.objective.valueOrDefault()
    return n + m - opt

def levenshteinDistance (a, b, matchFn, costFn) : 
    """
    Calculate the optimal tree edit distance between two lists,
    given the costs of matching two items on the list and 
    the costs of deleting an item in the optimal match.

    Example
    -------
    >>> skipSpace = lambda x : 0 if x == ' ' else 1
    >>> match = lambda a, b : 0 if a == b else 1
    >>> print(levenshteinDistance('sumitchaturvedi', 'sumi t ch at urvedi', match, skipSpace))

    In the above example, since we don't penalize for 
    deleting spaces, the levenshtein distance is 
    going to be 0.

    Parameters
    ----------
    a : list
        First string.
    b : list
        Second string.
    matchFn : lambda
        The cost of matching the ith character in a with 
        the jth character in b.
    costFn : lambda 
        The penalty of skipping over one character from 
        one of the strings.
    """
    n = len(a)
    m = len(b)
    row = list(range(n + 1))
    for j in range(1, m + 1) : 
        newRow = [0 for _ in range(n + 1)]
        for i in range(n + 1): 
            if i == 0 : 
                newRow[i] = j
            else : 
                match = matchFn(a[i-1], b[j-1])
                cost1 = costFn(a[i-1])
                cost2 = costFn(b[j-1])
                newRow[i] = min(row[i-1] + match, newRow[i-1] + cost1, row[i] + cost2)
        row = newRow
    return row[n]

def levelPathSetCuts (t, d):
    setNodeDepths(t)
    n = len(leaves(t))
    cuts = [list(t.nodes[n]['pathSet']) for n in t.nodes if t.nodes[n]['depth'] == d]
    singles = list(set(range(n)) - set(more_itertools.flatten(cuts)))
    singles = [[_] for _ in singles]
    cuts.extend(singles)
    clusterIds = np.zeros(n)
    for i, cluster in enumerate(cuts) :
        clusterIds[cluster] = i
    return clusterIds
