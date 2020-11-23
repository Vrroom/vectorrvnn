from functools import partial, reduce
import itertools
import more_itertools
import networkx as nx

def lca (t, a, b) : 
    r = findRoot(t)
    testNodes = set([a, b])
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

def setNodeDepths (t) :
    def dfs (n, p=None) : 
        if p is None : 
            t.nodes[n]['depth'] = 0
        else : 
            t.nodes[n]['depth'] = 1 + t.nodes[p]['depth']
        for c in t.neighbors(n) : 
            dfs(c, n)
    r = findRoot(t)
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
        Node whose descendents are to be computed.
    """
    neighbors = set(list(tree.neighbors(node)))
    descOfDesc = map(partial(descendants, tree), neighbors)
    descOfDesc = reduce(lambda a, b : a | b, descOfDesc, set())
    return {node} | neighbors | descOfDesc

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
    return T

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
    import pdb
    pdb.set_trace()
    import matplotlib.pyplot as plt
    plt.hist(things)
    plt.show()
