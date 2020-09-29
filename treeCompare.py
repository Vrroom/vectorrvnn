import numpy as np
import svgpathtools as svg
import random
from itertools import product
import networkx as nx
import more_itertools
import copy
from functools import reduce
from treeOps import findRoot, subtreeSize, descendants
from matching import bestAssignmentCost
from pulp import *

def ted (t1, t2) : 
    """
    Compute tree edit distance between two unordered 
    trees using Integer Programming. The cost of 
    insertion and deletion is 1. The cost of relabeling
    is 1 unless the pathSet's match exactly.

    Examples
    --------
    >>> from SVGData import SVGData
    >>> tree1 = SVGData('/Users/amaltaas/BTP/vectorrvnn/PartNetSubset/Train/10007.svg', "adjGraph", 10)
    >>> tree2 = SVGData('/Users/amaltaas/BTP/vectorrvnn/PartNetSubset/Train/1003.svg', "adjGraph", 10)
    >>> print(ted(tree1, tree2))
    """
    d = lambda x, y : 0 if t1.nodes[x]['pathSet'] == t2.nodes[y]['pathSet'] else 1
    n, m = t1.number_of_nodes(), t2.number_of_nodes()
    allPairs = list(product(range(n), range(m)))
    nodes1, nodes2 = list(t1.nodes), list(t2.nodes)
    v = np.array([[LpVariable(f'm_{x}_{y}', cat='Binary') for y in range(m)] for x in range(n)])
    distanceMatrix = np.array([[d(x, y) for y in nodes2] for x in nodes1])
    prob = LpProblem("TreeEditDistance", LpMinimize)
    prob += (v * (distanceMatrix - 2)).sum()
    for x in range(n) :
        prob += v[x,:].sum() <= 1
    for y in range(m) : 
        prob += v[:,y].sum() <= 1
    for (x, y), (x_, y_) in product(allPairs, allPairs) : 
        if x_ in descendants(t1, x) and y_ not in descendants(t2, y) : 
            prob += v[x, y] + v[x_, y_] <= 1
        if x_ not in descendants(t1, x) and y_ in descendants(t2, y) : 
            prob += v[x, y] + v[x_, y_] <= 1
    prob.solve()
    return value(prob.objective) + n + m

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

def levenshteinDistance (a, b, matchFn, costFn) : 
    """
    Calculate the optimal tree edit distance
    between two lists, given the costs of
    matching two items on the list and 
    the costs of deleting an item in the
    optimal match.

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
        The cost of matching the
        ith character in a with the 
        jth character in b.
    costFn : lambda 
        The penalty of skipping over
        one character from one of the
        strings.
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

def svgTreeEditDistance (t1, t2, paths, vbox) :
    """
    Compute two Tree objects (which capture)
    SVG document structure. Each node in the 
    tree is a group of paths. We have a 
    similarity function between these nodes 
    based on the levenshtein distance between the 
    path strings.
    
    Parameters
    ----------
    t1 : Data.Tree
        Tree one.
    t2 : Data.Tree
        Tree two.
    paths : list
        List of paths.
    vbox : list
        Bounding Box.
    """

    def pathMatchFn (a, b) : 

        def curveMatchFn (a, b) :
            """
            If the curve parameters match
            within a distance threshold, then
            we let them be.
            """
            if isinstance(a, type(b)) : 
                if isinstance(a, svg.Line) :
                    cond1 = abs(a.start - b.start) < thresh
                    cond2 = abs(a.end   - b.end  ) < thresh
                    return 0 if cond1 and cond2 else 1
                elif isinstance(a, svg.QuadraticBezier) : 
                    cond1 = abs(a.start   - b.start  ) < thresh
                    cond2 = abs(a.end     - b.end    ) < thresh
                    cond3 = abs(a.control - b.control) < thresh
                    return 0 if cond1 and cond2 and cond3 else 1
                elif isinstance(a, svg.CubicBezier) : 
                    cond1 = abs(a.start    - b.start   ) < thresh
                    cond2 = abs(a.end      - b.end     ) < thresh
                    cond3 = abs(a.control1 - b.control1) < thresh
                    cond4 = abs(a.control2 - b.control2) < thresh
                    return 0 if cond1 and cond2 and cond3 and cond4 else 1
                elif isinstance(a, svg.Arc) : 
                    cond1 = abs(a.start  - b.start ) < thresh
                    cond2 = abs(a.end    - b.end   ) < thresh
                    cond3 = abs(a.radius - b.radius) < thresh
                    return 0 if cond1 and cond2 and cond3 else 1
                else :
                    raise TypeError
            else :
                return 1

        def curveDelFn (a) : 
            return 1 
    
        if (a, b) not in cachedMatchVals : 
            pathA = paths[a].path
            pathB = paths[b].path
            maxLen = max(len(pathA), len(pathB))

            pathMatch = levenshteinDistance(pathA, pathB, curveMatchFn, curveDelFn)
            normalized = pathMatch / maxLen

            cachedMatchVals[(a, b)] = normalized
            cachedMatchVals[(b, a)] = normalized
        
        return cachedMatchVals[(a, b)]

    def pathDelFn (a) : 
        return 1

    def cost (u1, u2) :
        
        def pathSetDist (x, y) : 
            ps1 = t1.tree.nodes[x]['pathSet']
            ps2 = t2.tree.nodes[y]['pathSet']
            lev = levenshteinDistance(ps1, ps2, pathMatchFn, pathDelFn)
            maxLen = max(len(ps1), len(ps2))
            return lev / maxLen

        nbrs1 = list(t1.tree.neighbors(u1))
        nbrs2 = list(t2.tree.neighbors(u2))
        degree1 = t1.tree.out_degree(u1)
        degree2 = t2.tree.out_degree(u2)
        if degree1 == 0 and degree2 == 0 :
            return pathSetDist(u1, u2)
        elif degree1 == 0 : 
            return sub2[u2] - sub1[u1];
        elif degree2 == 0 :
            return sub1[u1] - sub2[u2]
        else :
            prod = list(product(nbrs1, nbrs2))
            costs = list(map(lambda x : cost(*x) + pathSetDist(*x), prod))
            nbrs2 = [str(_) for _ in nbrs2]
            costdict = dict(zip(prod, costs))
            return bestAssignmentCost(costdict)

    
    xmin, ymin, xmax, ymax = vbox
    diagonal = np.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
    thresh = diagonal / 10

    cachedMatchVals = dict() 
    sub1, sub2 = dict(), dict()
    subtreeSize(t1.root, t1.tree, sub1)
    subtreeSize(t2.root, t2.tree, sub2)
    c = cost(t1.root, t2.root)
    return c

def match (rt1, rt2) :
    """
    Given two full rooted binary trees,
    find the minimum nodes needed to be 
    added to either of the trees to make
    them isomorphic.

    >>> t1 = nx.DiGraph()
    >>> t2 = nx.DiGraph()
    >>> t1.add_edges_from([(0,1), (1,3), (0,2)])
    >>> t2.add_edges_from([(0,1), (0,2), (2,3)])
    >>> match((0, t1), (0, t2))

    Parameters
    ----------
    rt1 : tuple
        A (root index, nx.DiGraph) tuple representing
        the first rooted tree.
    rt2 : tuple
        The second rooted tree.
    """
    r1, t1 = rt1
    r2, t2 = rt2
    sub1 = dict()
    sub2 = dict()
    subsetsize(r1, t1, sub1)
    subsetsize(r2, t2, sub2)

    def cost (u1, u2) :
        nbrs1 = list(t1.neighbors(u1))
        nbrs2 = list(t2.neighbors(u2))
        degree1 = len(nbrs1)
        degree2 = len(nbrs2)
        if degree1 == 0 and degree2 == 0 :
            return 0
        elif degree1 == 0 : 
            return sub2[u2] - sub1[u1];
        elif degree2 == 0 :
            return sub1[u1] - sub2[u2]
        else :
            prod = list(product(nbrs1, nbrs2))
            costs = list(map(lambda x : cost(*x), prod))
            nbrs2 = [str(_) for _ in nbrs2]
            costdict = dict(zip(prod, costs))
            return bestAssignmentCost(costdict)

    return cost(r1, r2)
