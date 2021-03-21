from functools import partial, reduce, lru_cache
import itertools
import more_itertools
import networkx as nx
import numpy as np
from copy import deepcopy

def treeify (t) : 
    n = t.number_of_nodes()
    t_ = deepcopy(t)
    roots = [r for r in t.nodes if t.in_degree(r) == 0]
    if len(roots) > 1 : 
        edges = list(itertools.product([n], roots))
        t_.add_edges_from(edges)
        t_.nodes[n]['pathSet'] = leaves(t)
    return t_

def numNodes2Binarize (t) :
    """
    How many nodes should be added to 
    binarize a tree.
    """
    return sum([t.out_degree(n) - 2 for n in t.nodes if t.out_degree(n) > 2])

def lca (t, a, b, r) : 
    testNodes = {a, b}
    while True :
        desc = map(partial(descendants, t), t.neighbors(r))
        found = False
        for n in t.neighbors(r) : 
            desc = descendants(t, n)
            if testNodes.issubset(desc) : 
                r = n
                found = True
                break
        if not found : 
            break
    return r

@lru_cache(maxsize=128)
def lcaScore (t, a, b) : 
    if a == b : 
        return 0
    roots = [_ for _ in t.nodes if t.in_degree(_) == 0]
    subtrees = [descendants(t, _) for _ in roots]
    doesIntersect = list(map(lambda s : len(s.intersection({a, b})) == 2, subtrees))
    if any(doesIntersect) : 
        r = [r_ for r_, d in zip(roots, doesIntersect) if d].pop()
        l = lca(t, a, b, r)
        d = max(t.nodes[a]['depth'], t.nodes[b]['depth'])
        l_ = (d - t.nodes[l]['depth'])
        return l_
    return 5

def lofScore (t, a, b) :
    l = lca(t, a, b)
    sl = t.nodes[l]['subtree-size']
    sa = t.nodes[a]['subtree-size']
    sb = t.nodes[b]['subtree-size']
    return (sl - sa - sb) / sl

def setSubtreeSizes (t) : 
    def dfs (n) : 
        if t.out_degree(n) == 0 : 
            t.nodes[n]['subtree-size'] = 1
        else : 
            t.nodes[n]['subtree-size'] = sum(dfs(c) for c in t.neighbors(n)) + 1
        return t.nodes[n]['subtree-size']
    roots = [n for n in t.nodes if t.in_degree(n) == 0]
    for r in roots:
        dfs(r)

def setNodeDepths (t) :
    def dfs (n, p=None) : 
        if p is None : 
            t.nodes[n]['depth'] = 0
        else : 
            t.nodes[n]['depth'] = 1 + t.nodes[p]['depth']
        for c in t.neighbors(n) : 
            dfs(c, n)
    roots = [n for n in t.nodes if t.in_degree(n) == 0]
    for r in roots:
        dfs(r)

def setNodeBottomDepths (t) : 
    def dfs (n) : 
        if t.out_degree(n) == 0 : 
            t.nodes[n]['bottom-depth'] = 1
        else  :
            t.nodes[n]['bottom-depth'] = max(dfs(c) for c in t.neighbors(n)) + 1
        return t.nodes[n]['bottom-depth']
    roots = [n for n in t.nodes if t.in_degree(n) == 0]
    for r in roots:
        dfs(r)
    
def maxOutDegree (t) : 
    return max(t.out_degree(n) for n in t.nodes)

def maxDepth (t) :
    r = findRoot(t)
    def helper (n) : 
        if t.out_degree(n) == 0 : 
            return 1
        return 1 + max(map(helper, t.neighbors(n)))
    return helper(r)
    
def subtreeSize(s, t, subSize) :
    """
    Calculate the size of each
    subtree in the rooted tree t.

    Parameters
    ----------
    s : int
        The current vertex
    t : nx.DiGraph()
        The tree
    subSize : dict
        Store the size of the subtree
        rooted at each vertex.
    """
    subSize[s] = 1
    nbrs = list(t.neighbors(s))
    if t.out_degree(s) != 0 : 
        for u in nbrs :
            subtreeSize(u, t, subSize)
            subSize[s] += subSize[u]

def descendants (tree, node) : 
    """
    Find the set of descendants of this node 
    in the tree. 

    Include this node in the set as well.

    Parameters
    ----------
    tree : nx.DiGraph
        Rooted tree.
    node : any
        Node whose descendents are to be cpomputed.
    """
    if 'descendants' not in tree.nodes[node] : 
        neighbors = set(list(tree.neighbors(node)))
        descOfDesc = map(partial(descendants, tree), neighbors)
        descOfDesc = reduce(lambda a, b : a | b, descOfDesc, set())
        tree.nodes[node]['descendants'] =  {node} | neighbors | descOfDesc
    return tree.nodes[node]['descendants']

def leaves (tree) :
    """
    Returns the leaf nodes in a
    directed tree.

    Parameters
    ----------
    tree : nx.DiGraph
    """
    return list(filter (lambda x : tree.out_degree(x) == 0, tree.nodes))

def nonLeaves (tree) :
    """
    Returns the internal nodes in a
    directed tree.

    Parameters
    ----------
    tree : nx.DiGraph
    """
    return list(filter (lambda x : tree.out_degree(x) > 0, tree.nodes))

def findRoot (tree) :
    """
    Find the root of a tree.

    Parameters
    ----------
    tree : nx.DiGraph
        Rooted tree with unknown root.
    """
    return next(nx.topological_sort(tree))

def treeMap (T, r, function) : 
    """
    Apply a function on each node and accumulate
    results in a list.

    Parameters
    ----------
    T : nx.DiGraph
        The tree.
    r : object
        Root of the tree.
    function : lambda
        A function which takes as
        input the tree, current node 
        and performs some operation.
    """
    results = []
    for child in T.neighbors(r) :
        results.extend(treeMap(T, child, function))
    results.append(function(T, r, T.neighbors(r)))
    return results

def treeApplyRootFirst (T, r, function) :
    """
    Apply function to all nodes in the
    tree.

    Parameters
    ----------
    T : nx.DiGraph
        The tree.
    r : object
        Root of the tree.
    function : lambda
        A function which takes as
        input the tree, current node 
        and performs some operation.
    """
    function(T, r, T.neighbors(r))
    for child in T.neighbors(r) :
        treeApplyRootFirst(T, child, function)

def treeApplyChildrenFirst (T, r, function) : 
    """
    Apply function to all nodes in the
    tree.

    Parameters
    ----------
    T : nx.DiGraph
        The tree.
    r : object
        Root of the tree.
    function : lambda
        A function which takes as
        input the tree, current node 
        and performs some operation.
    """
    for child in T.neighbors(r) :
        treeApplyChildrenFirst(T, child, function)
    function(T, r, T.neighbors(r))

def mergeTrees (trees) : 
    """
    Merge a list of trees into a single 
    tree. 
    
    The catch is that the leaf nodes
    which represent path indices in our setup
    have to be distinctly labeled across the
    trees. 
    
    So we only relabel the internal
    nodes so that while composing, we
    don't introduce edges which shouldn't
    be there. 

    Finally we add a new root node with
    these subtrees as the children.

    Example
    -------
    >>> tree1 = nx.DiGraph()
    >>> tree2 = nx.DiGraph()
    >>> tree1.add_edges_from([(3, 1), (3, 2)])
    >>> tree2.add_edges_from([(4, 3), (5, 4)])
    >>> print(mergeTrees([tree1, tree2]).edges)

    Parameters
    ----------
    trees : list
    """
    def relabelTree (tree) : 
        nonlocal maxIdx
        internalNodes = nonLeaves(tree)
        newId = range(maxIdx, maxIdx + len(internalNodes))
        newLabels = dict(zip(internalNodes, newId))
        maxIdx += len(internalNodes)
        return nx.relabel_nodes(tree, newLabels, copy=True)

    allLeaves = list(more_itertools.collapse(map(leaves, trees)))
    maxIdx = max(allLeaves) + 1
    relabeledTrees = list(map(relabelTree, trees))
    roots = map(findRoot, relabeledTrees)
    newTree = nx.compose_all(relabeledTrees)
    newEdges = list(map(lambda x : (maxIdx, x), roots))
    newTree.add_edges_from(newEdges)
    newTree.nodes[maxIdx]['pathSet'] = allLeaves
    return newTree

def removeOneOutDegreeNodesFromTree (tree) : 
    """
    In many SVGs, it is the case that 
    there are unecessary groups which contain
    one a single group.

    In general these don't capture the hierarchy
    because the one out-degree nodes can be 
    removed without altering the grouping.

    Hence we have this function which removes
    one out-degree nodes from a networkx tree.

    Example
    -------
    >>> tree = nx.DiGraph()
    >>> tree.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (4, 6)])
    >>> tree = removeOneOutDegreeNodesFromTree(tree)
    >>> print(tree.number_of_nodes())

    Since the nodes 0 and 2 have out-degree 1,
    they'll be deleted and we'll be left
    with 5 nodes.
   
    Parameters
    ----------
    tree : nx.DiGraph
    """
    def remove (n) : 
        children = list(tree.neighbors(n))
        if tree.out_degree(n) == 1 : 
            child = children[0]
            tree.remove_node(n)
            return remove(child)
        else : 
            newEdges = list(itertools.product([n], map(remove, children)))
            tree.add_edges_from(newEdges)
            return n

    remove(findRoot(tree))
    topOrder = list(nx.topological_sort(tree))
    relabelDict = dict(zip(topOrder, range(tree.number_of_nodes())))
    tree = nx.relabel_nodes(tree, relabelDict)
    return tree

def treeFromNestedArray (nestedArray) :
    T = nx.DiGraph()
    while len(nestedArray) > 0 : 
        parent = nestedArray.pop()
        if isinstance(parent, int) :
            pathSet = [parent]
        else : 
            pathSet = list(more_itertools.collapse(parent))
            for child in parent : 
                T.add_edge(parent, child)
            nestedArray.extend(parent)
        T.nodes[parent]['pathSet'] = pathSet 
    # Relabel interior nodes.
    internalNodes = [n for n in T.nodes if T.out_degree(n) > 0]
    m = max(leaves(T)) + 1
    newLabels = range(m, m + len(internalNodes))
    newMapping = dict(zip(internalNodes, newLabels))
    T = nx.relabel_nodes(T, newMapping)
    assert nx.is_tree(T), "Didn't produce tree"
    return T

def pathInTree (T, x1, x2) : 
    """
    Return path from x1 to x2 in T.

    Assume that x1 is ancestor of x2.

    Parameters
    ----------
    T : nx.DiGraph
        Tree.
    x1 : node 
        Ancestor node in T.
    x2 : node
        Descendent node in T.

    Returns
    -------
    list of all nodes on path from x1 to x2
    including x1 and x2.
    """
    if x1 == x2 : 
        return [x1]
    c = next(filter(lambda x : x2 in descendants(T, x), T.neighbors(x1)))
    return [x1] + pathInTree(T, c, x2)

def leavesInSubtree (T, x) : 
    """
    Return leaves in subtree whose root is x.

    Parameters
    ----------
    T : nx.DiGraph
        Tree.
    x : node 
        Node in T.
    """
    return descendants(T, x) & set(leaves(T))

def computeLCAMatrix(T) : 
    """
    Compute LCA matrix where entries are 
    max-depth normalized LCA for each pair
    of nodes.

    Parameters
    ----------
    T : nx.DiGraph
        Tree.
    """
    n = T.number_of_nodes()
    T.lcaMatrix = np.zeros((n, n))
    T.maxDepth = maxDepth(T)
    setNodeDepths(T)
    for i, a in enumerate(T.nodes) : 
        for j, b in enumerate(T.nodes) : 
            T.lcaMatrix[i, j] = T.nodes[lca(T, a, b)]['depth'] / T.maxDepth

def parent(t, n) : 
    if t.in_degree(n) == 0 : 
        return None
    else : 
        return [p for p in t.nodes if n in t.neighbors(p)].pop()

def siblings(t, n): 
    p = parent(t, n)
    return set(t.neighbors(p)) - {n}

if __name__ == "__main__" : 
    import json
    from Dataset import SVGDataSet 
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load all the data
    trainDir = commonConfig['train_directory']
    trainData = SVGDataSet(trainDir, 'adjGraph', 10, useColor=False)
    things = list(map(maxDepth, trainData))
    setNodeDepths(trainData[0])
    import matplotlib.pyplot as plt
    plt.hist(things)
    plt.show()
