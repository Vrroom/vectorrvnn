""" Simple and conservative grouping heuristics """
import networkx as nx
from scipy.spatial import KDTree
from copy import deepcopy
import numpy as np
from vectorrvnn.utils.datastructures import *
from .descriptor import equiDistantPointsOnPolyline
import svgpathtools as svg

def fourier_descriptor (tree, i, nSamples=1000, freqs=20) :
    doc = tree.doc
    lines = deepcopy(tree.lines[i])
    lines.append(svg.Line(lines[-1].end, lines[0].start))
    pts = np.array(equiDistantPointsOnPolyline(doc, lines, nSamples, normalize=True)).T
    d2o = np.sqrt((pts * pts).sum(1))
    mi = argmin(d2o.tolist())
    pts = np.vstack((pts[mi:], pts[:(mi + 1)]))
    pts = pts - pts.mean(0)
    lens = np.sqrt((pts * pts).sum(1))
    an = np.fft.rfft(lens)
    hf = an[1:(freqs // 2) + 1]
    F = np.abs(np.hstack((hf.real, hf.imag)))
    F /= np.abs(an[0])
    return F

def centroid (tree, i, nSamples=1000) :
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
