""" https://diglib.eg.org/bitstream/handle/10.2312/egs20211016/029-032.pdf """
from sklearn.cluster import AgglomerativeClustering
from vectorrvnn.utils import *
from vectorrvnn.geometry import *
import networkx as nx
from more_itertools import collapse
from itertools import combinations, product
import random
from copy import deepcopy

AUTOGROUP = dict(
    color=autogroupColorSimilarity,
    stroke=autogroupStrokeSimilarity,
    shape=autogroupShapeHistogramSimilarity,
    place=autogroupPlacementDistance,
    area=autogroupAreaSimilarity
)

def _mergeTrees (r, containmentGraph, trees): 
    if containmentGraph.out_degree(r) == 0 : 
        return r
    else : 
        childClusters = dict(
            enumerate([
                _mergeTrees(c, containmentGraph, trees) 
                for c in containmentGraph.neighbors(r)
            ])
        )
        parenthesis = tree2Parenthesis(trees[r], symbols=childClusters)
        return (r, parenthesis)

def _descriptorWeightFn (lst, mat, variance) : 
    submat = mat[np.ix_(lst, lst)]
    submatRange = submat.max() - submat.min()
    return float(submatRange > 0.15 * variance)

def _dropExtraParents (graph) : 
    extra = [] 
    for i, j in product(graph.nodes, graph.nodes) : 
        if i != j : 
            ci = descendants(graph, i) - {i}
            cj = descendants(graph, j) - {j}
            if not isDisjoint(ci, cj) : 
                common = list(ci.intersection(cj))
                extra.extend([(i, _) for _ in common])
    graph_ = deepcopy(graph)
    graph_.remove_edges_from(extra)
    return graph_

def implies (p, q) : 
    """ p => q """
    return (p and q) or not p

def canMerge (i, j, occlusionGraph) : 
    """ 
    Two things can be merged if they don't occlude each other
    or if they do, they are adjacent in the z-index order
    """
    return implies(leqInPO(j, i, occlusionGraph), j == i + 1)\
            or implies(leqInPO(i, j, occlusionGraph), i == j + 1)

def autogroup (doc) : 
    paths = cachedPaths(doc)
    n = len(paths)
    # directed graph where a -> b iff a contains b
    containmentGraph = _dropExtraParents(
        simplifyPO(
            subgraph(
                relationshipGraph(doc, bboxContains, False),
                lambda x: x['bboxContains']
            )
        )
    )
    # find the similarity matrices for each function
    matrices = dictmap(
        lambda k, x: nx.to_numpy_matrix(
            relationshipGraph(
                doc, x, True, 
                containmentGraph=containmentGraph
            ),
            weight=x.__name__
        ),
        AUTOGROUP
    )
    # one of the matrices is a distance matrix. Normalize
    # and convert to similarity matrix.
    if matrices['place'].max() == 0 :
        matrices['place'][:, :] = 1
    else : 
        matrices['place'] = 1 - (matrices['place'] / matrices['place'].max())
    # calculate variance for unnormalized scores.
    variances = dictmap(lambda k, x: x.var(), matrices)
    m = reduce(lambda a, b: min(a.min(), b.min()), matrices.values())
    M = reduce(lambda a, b: max(a.max(), b.max()), matrices.values())
    normedMatrices = dictmap(
        lambda _, x: (x - m) / (M - m), 
        matrices
    )
    # Only siblings are considered for clustering.
    # Add a dummy node to all parent-less nodes.
    containmentGraph.add_edges_from(
        [(n, _) for _ in containmentGraph.nodes 
            if containmentGraph.in_degree(_) == 0])

    # partition node set into set of siblings
    parents = nonLeaves(containmentGraph)
    siblingSets = [list(containmentGraph.neighbors(p))
            for p in parents]
    # Find out appropriate weights for each band of siblings.
    weights = map(
        lambda x : dictmap(
            lambda k, v : _descriptorWeightFn(x, v, variances[k]),
            matrices
        ), 
        siblingSets
    )
    # Perform agglomerative clustering for each sibling set separately
    # after calculating the weights for that set.
    trees = dict()
    for p, x, wt in zip(parents, siblingSets, weights):  
        if len(x) == 1 : 
            trees[p] = nx.DiGraph()
            trees[p].add_node(0)
            continue
        totalWt = sum(wt.values())
        submats = dictmap(lambda k, v: v[np.ix_(x, x)], normedMatrices)
        weightedSubmats = map(lambda x, y : x * y, submats.values(), wt.values())
        # this hack is for when totalWt = 0 in which case M = 1
        M = (sum(weightedSubmats) + 1e-4) / (totalWt + 1e-4)
        agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
        agg.fit(1 - M)
        t = hac2nxDiGraph(list(range(len(x))), agg.children_)
        trees[p] = t

    nestedArray = [_mergeTrees(n, containmentGraph, trees)[1]]
    return treeFromNestedArray(nestedArray)

