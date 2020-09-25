import numpy as np
import random
import more_itertools
import copy
from functools import reduce
from treeOps import findRoot

def treeKCut (tree, k) :
    """
    Given a tree, make k cuts. Ideally the k cuts 
    should give evenly sized sets.

    Parameters
    ----------
    tree : Tree
        Hierachical clustering from which
        k evenly sized sets.
    """
    def selector (a, b) : 
        if a['level'] < b['level'] : 
            return a
        elif a['level'] > b['level'] :
            return b
        elif len(a['ids']) > len(b['ids']) :
            return a
        else :
            return b

    def split (T, best) :
        level, arr = None, None
        if len(best['ids']) > 1 :
            arr = copy.deepcopy(best['ids'])
            level = best['level']
        else :
            i = best['ids'][0]
            arr = list(T.neighbors(i))
            level = best['level'] + 1
        index = random.randint(0, len(arr) - 1)
        left = { 'level':level, 'ids':arr[:index] + arr[index+1:]}
        right = { 'level': level, 'ids':[arr[index]] }
        return left, right

    T = tree.tree
    r = findRoot(T)
    candidates = [{'level': 0, 'ids': [r]}]
    leaves = [];
    while len(candidates) + len(leaves) < k :
        best = reduce(selector, candidates)
        candidates.remove(best)
        left, right = split (T, best)
        if len(left['ids']) > 1 or T.out_degree(left['ids'][0]) > 0 :
            candidates.append(left)
        else :
            leaves.append(left)

        if len(right['ids']) > 1 or T.out_degree(right['ids'][0]) > 0 :
            candidates.append(right)
        else :
            leaves.append(right)

    candidates.extend(leaves)
    cuts = list(map(lambda c : list(
            more_itertools.flatten(
                map(lambda i : T.nodes[i]['pathSet'], c['ids'])))
            , candidates))
    return cuts

def hierarchicalClusterCompareFM (t1, t2, K) : 
    """
    Implementation of: 

        A Method for Comparing Two Hierarchical Clusterings

    This method gives statistics using which it can be
    decided whether two hierarchical clusterings are 
    similar.

    FM stands for Fowlkes and Mallows.

    Parameters
    ----------
    t1 : Tree
        Tree one.
    t2 : Tree
        Tree two.
    K : Upper bound on cuts.
    """
    assert t1.nPaths == t2.nPaths
    n = t1.nPaths
    bs = []
    es = [] 
    for k in range(2, K): 
        cuts1 = treeKCut(t1, k)
        cuts2 = treeKCut(t2, k)
        M = np.zeros((k, k))
        for i, ci in enumerate(cuts1) : 
            for j, cj in enumerate(cuts2) :
                M[i, j] = len(set(ci) & set(cj))
        tk = (M * M).sum() - n
        mi, mj = M.sum(axis=0), M.sum(axis=1)
        pk = (mi ** 2).sum() - n
        qk = (mj ** 2).sum() - n
        bk = tk / np.sqrt(pk * qk)
        ek = np.sqrt(pk * qk) / (n * (n - 1))
        bs.append(bk)
        es.append(ek)
    return np.array(bs)
