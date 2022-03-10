""" Simple and conservative grouping heuristics """
import networkx as nx
from scipy.spatial import KDTree
import numpy as np
from vectorrvnn.utils.datastructures import *
from .descriptor import equiDistantPointsOnPolyline

def fourier_descriptor (tree, i, nSamples=1000, freqs=20) :
    doc = tree.doc
    lines = tree.lines[i]
    pts = np.array(equiDistantPointsOnPolyline(doc, lines, nSamples, normalize=True)).T
    d2o = np.sqrt((pts * pts).sum(1))
    mi = argmin(d2o.tolist())
    pts = np.vstack((pts[mi:], pts[:(mi + 1)]))
    pts = pts - pts.mean(0)
    pts = np.sqrt((pts * pts).sum(1))
    an = np.fft.rfft(pts)
    pos = an[1:freqs//4 + 1]
    neg = an[-freqs//4:]
    an = np.hstack([pos, neg])
    an = np.hstack([an.real, an.imag])
    return an

def centroid (tree, i, nSamples=100) :
    doc = tree.doc
    lines = tree.lines[i]
    pts = np.array(equiDistantPointsOnPolyline(doc, lines, nSamples, normalize=True)).T
    return pts.mean(0)

def maximalCliques (tree, featureFns, margin) :
    """
    Group paths using maximal cliques

    1. Compute features for paths.
    2. Find all pairs of paths whose features are within 
    `margin` distance of each other
    3. Construct a graph from these pairs.
    4. Report maximal cliques

    Example
    -------
    >>> tree = SVGData('file.svg')
    >>> maximalCliques(tree, [fourier_descriptor, centroid], 1e-2)

    This should group strokes and fills. These commonly occur
    as separate paths in graphics.
    """
    fs = []
    for i in range(tree.nPaths): 
        fs.append(np.hstack([fn(tree, i) for fn in featureFns]))
    fs = np.array(fs)
    kdtree = KDTree(fs) 
    pairs = kdtree.query_pairs(margin)
    G = nx.Graph()
    G.add_edges_from(pairs)
    return list(nx.find_cliques(G))
