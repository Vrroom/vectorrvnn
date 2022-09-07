""" Simple and conservative grouping heuristics """
import networkx as nx
from scipy.spatial import KDTree
from copy import deepcopy
import numpy as np
from vectorrvnn.utils.datastructures import *
from vectorrvnn.utils.bipartite import *
from itertools import product
from .descriptor import equiDistantPointsOnPolyline
import svgpathtools as svg

def fourier_descriptor (tree, i, nSamples=500, freqs=20) :
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
    F /= np.abs(F).max()
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

def shapeContexts (tree, i, nSamples=50) :
    """ Give a list of shape contexts that can be used for matching """
    doc, lines = tree.doc, tree.lines[i]
    pts = np.array(equiDistantPointsOnPolyline(doc, lines, nSamples, normalize=True)).T
    grid = pts.reshape((-1, 1, 2))
    grid = np.repeat(grid, nSamples, axis=1)
    diffs = grid - pts
    mask = np.eye(nSamples).astype(bool)
    diffs = diffs[~mask].reshape(nSamples, nSamples - 1, 2)
    logNorms = np.log2(np.linalg.norm(diffs, axis=2) + 1e-7)
    thetas = np.arctan2(diffs[:, :, 0], diffs[:, :, 1])
    xbins = np.linspace(-10, 0.5, 10)
    ybins = np.linspace(-np.pi, np.pi, 10)
    contexts = []
    # Figure out how to vectorize this step.
    for i in range(nSamples) :
        H, *_ = np.histogram2d(logNorms[i], thetas[i], bins=[xbins, ybins])
        H = H / (H.sum() + 1e-6)
        contexts.append(H.ravel())
    return np.stack(contexts)

def shapeContextLoss (A, B) :
    n = A.shape[0]
    A_ = np.tile(np.expand_dims(A, 1), (1, n, 1))
    B_ = np.tile(np.expand_dims(B, 0), (n, 1, 1))
    D = (0.5 * (((A_ - B_) ** 2) / (A_ + B_ + 1e-5))).sum(2)
    costDict = dict()
    for i, j in product(range(n), range(n)) :
        costDict[(i, j)] = D[i, j]
    matching = optimalBipartiteMatching(costDict)
    costs = []
    for i, j in matching.items() :
        costs.append(costDict[(i, j)])
    return np.median(costs)

def groupByShapeContexts (tree) :
    plist = range(tree.nPaths)
    scs = [shapeContexts(tree, i) for i in plist] 
    ces = np.stack([centroid(tree, i) for i in plist])
    kdtree = KDTree(ces) 
    groups = set()
    for i in range(tree.nPaths) : 
        ce = ces[i]
        nbrs = kdtree.query_ball_point(ce, 0.2) 
        for j in nbrs : 
            iou = tree.bbox[i].iou(tree.bbox[j])
            if j != i and iou > 0.25 and shapeContextLoss(scs[i], scs[j]) < 0.4 :
                if not ((i, j) in groups or (j, i) in groups): 
                    groups.add((i, j))
    return list(groups)
