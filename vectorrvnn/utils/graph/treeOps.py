from functools import partial, reduce, lru_cache
import itertools
import more_itertools
import networkx as nx
import numpy as np
from copy import deepcopy
from vectorrvnn.utils import argmax

def treeUnion (t1, t2) : 
    """
    Take the union of two trees.

    Since these trees actually represent graphic organizations,
    the leaves in both trees will be labelled by their render
    order. When we combine both the graphics, the paths from 
    the second are pushed on top of the first in the render order.
    For this purpose, we need to relabel the leaves in the second
    tree. As a result the non leaf nodes in both trees also get
    relabelled.
    """
    l1, l2 = len(leaves(t1)), len(leaves(t2))
    nl1, nl2 = len(nonLeaves(t1)), len(nonLeaves(t2))
    # Find the mapping in the combination
    t1NonLeafMapping = dict([(n, l1 + l2 + i) for i, n in enumerate(nonLeaves(t1))])
    t2LeafMapping = dict([(n, l1 + n) for n in leaves(t2)])
    t2NonLeafMapping = dict([(n, l1 + l2 + nl1 + i) for i, n in enumerate(nonLeaves(t2))])
    # relabel
    t1_ = nx.relabel_nodes(t1, t1NonLeafMapping)
    t2_ = nx.relabel_nodes(t2, {**t2LeafMapping, **t2NonLeafMapping})
    # take union
    union = forest2tree(nx.union(t1_, t2_))
    for n in union.nodes : 
        keys = list(union.nodes[n].keys())
        for k in keys : 
            union.nodes[n].pop(k, None)
    return union

def trimTreeByDepth (t, levels) : 
    """ 
    Remove all nodes lower than a certain depth.

    A copy of the original tree is returned
    """
    t_ = deepcopy(t)
    setNodeDepths(t_)
    removeNodes = [n for n in t_.nodes 
            if t_.nodes[n]['depth'] > levels]
    t_.remove_nodes_from(removeNodes)
    return t_

def forest2tree (t) : 
    """ convert a forest into a single tree """
    n = t.number_of_nodes()
    t_ = deepcopy(t)
    roots = [r for r in t.nodes if t.in_degree(r) == 0]
    if len(roots) > 1 : 
        edges = list(itertools.product([n], roots))
        t_.add_edges_from(edges)
        t_.nodes[n]['pathSet'] = leaves(t)
    return t_

def numNodes2Binarize (t) :
    """ How many nodes should be added to binarize a tree  """
    return sum([t.out_degree(n) - 2 for n in t.nodes if t.out_degree(n) > 2])

def lca (t, a, b) : 
    """ brute force computation for lowest common ancestor """
    setNodeDepths(t)
    test = {a, b}
    r = findRoot(t) 
    anc = list(filter(
        lambda x : test.issubset(descendants(t, x)),
        t.nodes
    ))
    ancDepths = [t.nodes[n]['depth'] for n in anc]
    return anc[argmax(ancDepths)]

@lru_cache(maxsize=128)
def lcaScore (t, a, b) : 
    if a == b : 
        return 0
    l = lca(t, a, b)
    d = max(t.nodes[a]['depth'], t.nodes[b]['depth'])
    return (d - t.nodes[l]['depth'])

def distanceInTree (t, a, b) : 
    """ Find the distance between two nodes in a tree """
    l = lca(t, a, b) 
    aDepth = t.nodes[a]['depth']
    bDepth = t.nodes[b]['depth']
    lDepth = t.nodes[l]['depth']
    return (aDepth - lDepth) + (bDepth - lDepth)

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
    Calculate the size of each subtree in the rooted tree t.

    Parameters
    ----------
    s : int
        The current vertex
    t : nx.DiGraph
        The tree
    subSize : dict
        Store the size of the subtree rooted at each vertex.
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
    return sorted(list(filter(lambda x: tree.out_degree(x) == 0, tree.nodes)))

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
    Apply a function on each node and accumulate results in a list.

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
    Apply function to all nodes in the tree.

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
    Merge a list of trees into a single tree. 
    
    The catch is that the leaf nodes which represent path indices 
    in our setup have to be distinctly labeled across the trees. 
    
    So we only relabel the internal nodes so that while composing, we
    don't introduce edges which shouldn't be there. 

    Finally we add a new root node with these subtrees as the children.

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
        newLabels = dict(zip(internalNodes, itertools.count(maxIdx)))
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
    In many SVGs, it is the case that there are unecessary groups 
    which contain one a single group.

    In general these don't capture the hierarchy because the one 
    out-degree nodes can be removed without altering the grouping.

    Hence we have this function which removes one out-degree nodes 
    from a networkx tree.

    Example
    -------
    >>> tree = nx.DiGraph()
    >>> tree.add_edges_from([(0, 1), (1, 2), (1, 3), (2, 4), (4, 5), (4, 6)])
    >>> tree = removeOneOutDegreeNodesFromTree(tree)
    >>> print(tree.number_of_nodes())

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
    # Leave the leaf nodes alone and relabel the internal nodes consecutively
    beg = max(leaves(tree)) + 1
    internalRelabel = dict(zip(nonLeaves(tree), itertools.count(beg)))
    tree = nx.relabel_nodes(tree, internalRelabel)
    return tree

def parentheses2tree (par) :
    """
    Convert a nested tuple into a DiGraph

    nestedArray := int | tuple of ints

    All tuples are internal nodes and ints are leaf nodes. 
    """
    if isinstance(par, int) : 
        T = nx.DiGraph()
        T.add_node(par)
        T.nodes[par]['pathSet'] = (par,) 
    else : 
        trees = list(map(parentheses2tree, par))
        T = mergeTrees(trees)
    assert nx.is_tree(T), "Didn't produce tree"
    return T

def tree2parentheses (tree, symbols, r=None) :
    """ 
    Inverse of the above method (when symbols is an identity map)
    """
    if r is None: 
        r = findRoot(tree)
    if tree.out_degree(r) == 0 : 
        return symbols[r]
    else : 
        return tuple([
            tree2parentheses(tree, symbols, r=_) 
            for _ in tree.neighbors(r)
        ])

def pathInTree (T, x1, x2) : 
    """
    Return path from x1 to x2 in T, assuming x1 is ancestor of x2.

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

def parents(t, n) : 
    """ Find parents of node n in dag t """
    return [p for p in t.nodes if n in t.neighbors(p)]

def parent(t, n) : 
    """ Find parent of node n in tree t """ 
    if t.in_degree(n) == 0 : 
        return None
    else : 
        return [p for p in t.nodes if n in t.neighbors(p)].pop()

def siblings(t, n): 
    """ Find siblings of node n in tree """
    p = parent(t, n)
    if p is None: 
        return {}
    return set(t.neighbors(p)) - {n}

def maximumNodesInAnyLevel (tree) : 
    """ Find the thickest level in the tree """
    setNodeDepths(tree)
    M = maxDepth(tree)
    return max([
        len([_ for _ in tree.nodes if tree.nodes[_]['depth'] == i])
        for i in range(M)
    ])
