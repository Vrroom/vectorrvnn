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
from vectorrvnn.utils.bipartite import *
from collections import defaultdict

def ted (t1, t2, matching=False) :
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
        row and col are the names used to index the dataframe.
        """
        var = [[LpVariable(f'm_{x}_{y}', cat='Binary') for y in col] for x in row]
        return pd.DataFrame(data=var, columns=col, index=row)

    def computeW (x, y) :
        """
        Compute W matrix recursively and memoize results.
        """
        nonlocal W, mStars
        if np.isinf(W.loc[x, y]) :
            if t1.out_degree(x) == 0 or t2.out_degree(y) == 0 :
                W.loc[x, y] = 2 - d(x, y)
                mStars[(x, y)] = []
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
                mStars[(x, y)] = list(filter(lambda p: v.loc[p[0], p[1]].varValue == 1, product(set1, set2)))
                W.loc[x, y] = prob.objective.valueOrDefault()
        return W.loc[x, y]

    def matchMatrix () :
        topMatches = list(filter(lambda p: v.loc[p[0], p[1]].varValue == 1, product(t1.nodes, t2.nodes)))
        mPairs = []
        t1Map = dict(map(reversed, enumerate(t1.nodes)))
        t2Map = dict(map(reversed, enumerate(t2.nodes)))
        while len(topMatches) > 0 :
            p = topMatches.pop()
            mPairs.append(p)
            topMatches.extend(mStars[p])
        mm = np.zeros((n, m), dtype=int)
        for ti, tj in mPairs:
            mm[t1Map[ti], t2Map[tj]] = 1
        return mm

    n, m = t1.number_of_nodes(), t2.number_of_nodes()
    mStars = dict()
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
    if matching : 
        return n + m - opt, matchMatrix()
    else : 
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

def fmi (t, t_, level) : 
    return metrics.fowlkes_mallows_score(
        _levelPathSetCuts(t , level),
        _levelPathSetCuts(t_, level)
    )

def _levelPathSetCuts (t, d):
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

def cted (t1, t2, matching=False): 
    """ 
    cted stands for constrained tree edit distance. See

        https://link.springer.com/content/pdf/10.1007/BF01975866.pdf

    for more information.
    """
    def d (x, y) :
        """
        Cost of relabeling node x with node y is
        the symmetric difference of the pathSets divided
        by their union.
        """
        return 0 if x == y else 1
        ps1 = set(t1.nodes[x]['pathSet'])
        ps2 = set(t2.nodes[y]['pathSet'])
        return len(ps1 ^ ps2) / len(ps1 | ps2)

    tTable, fTable = dict(), dict()
    # set up base cases: 
    # cost for matching empty trees
    tTable[(None, None)] = fTable[(None, None)] = (0, [])
    # cost for matching tree with empty tree
    for n in t1.nodes : 
        tTable[(n, None)] = (len(descendants(t1, n)) , [])
        fTable[(n, None)] = (tTable[(n, None)][0] - 1, [])
    # cost for matching empty tree with tree
    for m in t2.nodes : 
        tTable[(None, m)] = (len(descendants(t2, m)) , [])
        fTable[(None, m)] = (tTable[(None, m)][0] - 1, [])

    def FTable(i, j) : 
        nonlocal fTable
        if (i, j) in fTable :
            return fTable[(i, j)]
        elif t1.out_degree(i) == 0 : 
            return fTable[(None, j)]
        elif t2.out_degree(j) == 0 : 
            return fTable[(i, None)]
        deleteBoth = fTable[(None, j)][0] + fTable[(i, None)][0]
        best1, min1 = [], 0
        for y in t2.neighbors(j) : 
            c, matches = FTable(i, y)
            cost = c - fTable[(None, y)][0] - fTable[(i, None)][0]
            if cost < min1 : 
                best1 = matches
                min1  = cost
        case1 = deleteBoth + min1
        best2, min2 = [], 0
        for x in t1.neighbors(i) : 
            c, matches = FTable(x, j)
            cost = c - fTable[(x, None)][0] - fTable[(None, j)][0]
            if cost < min2 : 
                best2 = matches
                min2 = cost
        case2 = deleteBoth + min2
        A = [f'a_{_}' for _ in t2.neighbors(j)]
        B = [f'b_{_}' for _ in t1.neighbors(i)]
        subCallTable = dict()
        for p in product(t1.neighbors(i), t2.neighbors(j)) : 
            subCallTable[p] = TTable(*p)
        for p in product(t1.neighbors(i), B) : 
            subCallTable[p] = tTable[(p[0], None)]
        for p in product(A, t2.neighbors(j)) : 
            subCallTable[p] = tTable[(None, p[1])]
        for p in product(A, B) : 
            subCallTable[p] = (0, [])
        costTable = dictmap(lambda k, v: v[0], subCallTable)
        matching = optimalBipartiteMatching(costTable)
        case3 = sum([costTable[e] for e in matching.items()])
        minCost = min(case1, case2, case3) 
        best = []
        if minCost == case1 : 
            best = best1
        elif minCost == case2 : 
            best = best2
        else : 
            for a in t1.neighbors(i) : 
                best += subCallTable[(a, matching[a])][1]
        fTable[(i, j)] = (minCost, best)
        return minCost, best
            
    def TTable (i, j) : 
        nonlocal tTable
        if (i, j) in tTable : 
            return tTable[(i, j)]
        deleteBoth = tTable[(None, j)][0] + tTable[(i, None)][0]
        best1, min1 = [], 0
        for y in t2.neighbors(j) : 
            c, matches = TTable(i, y)
            cost = c - tTable[(None, y)][0] - tTable[(i, None)][0]
            if cost < min1 : 
                best1 = matches
                min1 = cost
        case1 = deleteBoth + min1
        best2, min2 = [], 0
        for x in t1.neighbors(i) : 
            c, matches = TTable(x, j)
            cost = c - tTable[(x, None)][0] - tTable[(None, j)][0]
            if cost < min2 : 
                best2 = matches
                min2 = cost
        case2 = deleteBoth + min2
        c, fmatches = FTable(i, j)
        case3 = c + d(i, j) 
        minCost = min(case1, case2, case3)
        best = []
        if case1 == minCost: 
            best = best1
        elif case2 == minCost : 
            best = best2
        else : 
            best = [(i, j)] + fmatches
        tTable[(i, j)] = (minCost, best)
        return minCost, best

    def matchMatrix () :
        t1Map = dict(map(reversed, enumerate(t1.nodes)))
        t2Map = dict(map(reversed, enumerate(t2.nodes)))
        n, m = len(t1.nodes), len(t2.nodes)
        mm = np.zeros((n, m), dtype=int)
        for ti, tj in matches:
            mm[t1Map[ti], t2Map[tj]] = 1
        return mm
            
    opt, matches = TTable(findRoot(t1), findRoot(t2))
    if matching : 
        return opt, matchMatrix()
    else : 
        return opt
