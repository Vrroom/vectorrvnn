""" suggero - https://mi-lab.org/files/2013/10/suggero.pdf """
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from vectorrvnn.geometry import *
from vectorrvnn.utils.svg import cachedPaths, withoutDegeneratePaths
from vectorrvnn.utils.graph import hac2nxDiGraph
import networkx as nx
from copy import deepcopy

USE_LITE = True

SUGGERO_ADVANCED = [localProximity, globalProximity, 
        fourierDescriptorDistance, fillDistance, 
        strokeDistance, strokeWidthDifference, 
        endpointDistance, parallelismDistance]
SUGGERO_LITE = SUGGERO_ADVANCED[:-2]

def affinityMatrix (doc, affinityFn) :
    G = relationshipGraph(doc, affinityFn, True)
    M = nx.to_numpy_matrix(G, weight=affinityFn.__name__)
    M = M / (M.max() + 1e-5)
    return M

def combinedAffinityMatrix (doc, affinityFns, weights) : 
    affinityMatrices = [affinityMatrix(doc, fn) for fn in affinityFns]
    combinedMatrix = sum(w * M for w, M in zip(weights, affinityMatrices))
    return combinedMatrix

def suggero (tree, subtrees=None) : 
    doc = tree.doc
    paths = cachedPaths(doc)
    if subtrees is None : 
        subtrees = leaves(tree)
    subtrees = deepcopy(subtrees)
    paths = [paths[i] for i in subtrees]
    affinityFns = SUGGERO_LITE if USE_LITE else SUGGERO_ADVANCED
    nfns = len(affinityFns)
    uniformWts = (1 / nfns) * np.ones(nfns)
    M = combinedAffinityMatrix(doc, affinityFns, uniformWts)
    M = M[:, subtrees][subtrees, :]
    agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
    agg.fit(M)
    cpy = deepcopy(tree)
    cpy.initTree(hac2nxDiGraph(subtrees, agg.children_))
    return cpy
