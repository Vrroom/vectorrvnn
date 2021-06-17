""" distance functions for comparing paths """
import re
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import *
import itertools
from functools import lru_cache
from collections import namedtuple
from skimage import color
import numpy as np
from .descriptor import *
from .utils import *
from .boxes import *
from vectorrvnn.utils import *

def _getPathPair(doc, i, j) :
    paths = cachedPaths(doc)
    return paths[i], paths[j]

def circleIoU (circle1, circle2) : 
    """ intersection over union for circles """
    intersectionArea = circlesIntersectionArea(circle1, circle2)
    unionArea = circleArea(circle1) + circleArea(circle2) - intersectionArea
    return intersectionArea / unionArea

def hausdorff (a, b) : 
    """ symmetric hausdorff distance between point clouds """
    return max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])

def localProximity(doc, i, j, **kwargs) : 
    """ 
    Local proximity between paths as the 
    distance between center of bounding boxes.
    """
    path1, path2 = _getPathPair(doc, i, j)
    center1 = pathBBox(path1.path).center()
    center2 = pathBBox(path2.path).center()
    sqDist = complexDot(center1 - center2, center1 - center2)
    return np.sqrt(sqDist)

def globalProximity(doc, i, j, **kwargs) :
    """ 
    Global proximity between paths as the 
    distance as the average distance between points
    on the paths.
    """
    samples1 = equiDistantSamples(doc, i, 
            nSamples=10, normalize=False)
    samples2 = equiDistantSamples(doc, j, 
            nSamples=10, normalize=False)
    samples1 = np.array(samples1)
    samples2 = np.array(samples2)
    return np.linalg.norm(samples1 - samples2, axis=0).mean()

def fourierDescriptorDistance (doc, i, j, **kwargs) : 
    """ 
    Distance between fourier descriptors to 
    measure curve similarity.
    """
    return np.linalg.norm(fd(doc, i) - fd(doc, j))

def fillDistance (doc, i, j, **kwargs) : 
    """ distance between fill color in CIELAB space """
    path1, path2 = _getPathPair(doc, i, j)
    lab1 = color.rgb2lab(pathColor(path1, 'fill'))
    lab2 = color.rgb2lab(pathColor(path2, 'fill'))
    return np.linalg.norm(lab1 - lab2)

def strokeDistance (doc, i, j, **kwargs) : 
    """ distance between stroke color in CIELAB space """
    path1, path2 = _getPathPair(doc, i, j)
    lab1 = color.rgb2lab(pathColor(path1, 'stroke'))
    lab2 = color.rgb2lab(pathColor(path2, 'stroke'))
    return np.linalg.norm(lab1 - lab2)

def strokeWidthDifference (doc, i, j, **kwargs) : 
    """ absolute difference in stroke width """
    path1, path2 = _getPathPair(doc, i, j)
    return abs(pathStrokeWidth(path1) - pathStrokeWidth(path2))

@lru_cache(maxsize=128)
def _pathsWithTolerance (doc) : 
    """ 
    Set tolerance distance on document paths. 
    If two path endpoints are within this distance
    then it is likely that they should be 
    continuation of one another
    """
    newPaths = []
    paths = cachedPaths(doc)
    for path in paths:  
        FlattenedPath = namedtuple('FlattenedPath', (*path._fields, 'tol'))
        start, end = path.path.start, path.path.end
        start = np.array([start.real, start.imag])
        end = np.array([end.real, end.imag])
        tol1 = min(avgDistancePoint2Path(start, doc, i) for i, _ in enumerate(paths))
        tol2 = min(avgDistancePoint2Path(end, doc, i) for i, _ in enumerate(paths))
        tol = min(tol1, tol2)
        newPaths.append(FlattenedPath(*path, tol=tol))
    return newPaths

def endpointDistance (doc, i, j, **kwargs) :
    """
    Distance is low if two paths' endpoints can
    be merged. High otherwise.
    """
    paths = _pathsWithTolerance(doc)
    path1, path2 = paths[i], paths[j]
    r1, r2 = path1.tol, path2.tol
    c1 = (r1, path1.path.start)
    c2 = (r1, path1.path.end)
    c3 = (r2, path2.path.start)
    c4 = (r2, path2.path.end)
    score = 1 - max(
            circleIoU(c1, c3), 
            circleIoU(c1, c4), 
            circleIoU(c2, c3), 
            circleIoU(c2, c4))
    return score

def isometricDistance (doc, i, j, **kwargs) : 
    """ 
    Get the relative error in the best isometric
    transformation from i to j. 
    """
    path1, path2 = _getPathPair(doc, i, j)
    _, _, _, relE = isometry(path1.path, path2.path)
    return relE

def parallelismDistance (doc, i, j, **kwargs) : 
    """ try to figure out if the paths are parallel """
    path1, path2 = _getPathPair(doc, i, j) 
    l1, l2 = path1.path.length(), path2.path.length()
    samplingDistance = 0.02 * min(l1, l2)
    n = int(l1 / samplingDistance)
    m = int(l2 / samplingDistance)
    samples1 = np.array(equiDistantSamples(doc, i, nSamples=n, normalize=False)).T
    samples2 = np.array(equiDistantSamples(doc, j, nSamples=m, normalize=False)).T
    if m < n : 
        n, m = m, n
        samples1, samples2 = samples2, samples1
    if n / m < 0.1 : 
        return 10
    strokeLen = samplingDistance * n 
    E = []
    stride = max(1, (m - n + 1) // 10)
    for i in range(0, m - n + 1, stride) :
        seg1 = samples1 + (1e-3 * np.random.randn(n, 2))
        seg2 = samples2[i:i+n] + (1e-3 * np.random.randn(n, 2))
        R, t, e = optimalRotationAndTranslation(seg1, seg2)
        theta = abs(np.arctan(R[1, 0] / (R[0, 0] + 1e-3)) * 2 / np.pi)
        B = 1
        A, _ = -np.polyfit(seg1[:, 0], seg1[:, 1], deg=1)
        N_ = np.array([A, B]).T @ R
        d = distanceOfPointFromLine(seg1[0], (seg2[0], N_.T)) / strokeLen
        E.append((theta + e + d) / 3)
    return min(E)

def areaIntersectionDistance(doc, i, j, **kwargs) : 
    """ find the area of intersection of two paths """
    imi, imj = pathBitmap(doc, i, fill=False), pathBitmap(doc, j, fill=False)
    return (imi * imj).sum()

def autogroupAreaSimilarity (doc, i, j, **kwargs) :
    imi, imj = pathBitmap(doc, i), pathBitmap(doc, j)
    a1 = (imi > 0).sum()
    a2 = (imj > 0).sum()
    return 1 - abs(a1 - a2) / (max(a1, a2) + 1e-6)

def autogroupPlacementDistance (doc, i, j, **kwargs) : 
    path1, path2 = _getPathPair(doc, i, j)
    l1, l2 = path1.path.length(), path2.path.length()
    samplingDistance = 0.02 * min(l1, l2)
    n = int(l1 / samplingDistance)
    m = int(l2 / samplingDistance)
    samples1 = list(zip(*equiDistantSamples(doc, i, 
        nSamples=n, normalize=False)))
    samples2 = list(zip(*equiDistantSamples(doc, j, 
        nSamples=m, normalize=False)))
    ls1 = LineString(samples1)
    ls2 = LineString(samples2)
    return ls1.distance(ls2)

def autogroupShapeHistogramSimilarity (doc, i, j, **kwargs) : 
    """
    Reference : 
        https://diglib.eg.org/bitstream/handle/10.2312/egs20211016/short1005_supp.pdf?sequence=2&isAllowed=y
    """
    a, b = shapeHistogram(doc, i), shapeHistogram(doc, j)
    return histScore(a, b)

def autogroupStrokeSimilarity (doc, i, j, **kwargs) : 
    """
    Reference : 
        https://diglib.eg.org/bitstream/handle/10.2312/egs20211016/short1005_supp.pdf?sequence=2&isAllowed=y
    """
    path1, path2 = _getPathPair(doc, i, j) 

    c1 = pathColor(path1, 'stroke')
    c2 = pathColor(path2, 'stroke')
    colorScore = normalizedCiede2000Score(c1, c2)

    sw1, sw2 = pathStrokeWidth(path1), pathStrokeWidth(path2)
    swScore = 1 - abs(sw1 - sw2) / (max(sw1, sw2) + 1e-5)

    lc1, lc2 = pathStrokeLineCap(path1), pathStrokeLineCap(path2)
    lcScore = 1 if lc1 == lc2 else 0

    lj1, lj2 = pathStrokeLineJoin(path1), pathStrokeLineJoin(path2)
    ljScore = 1 if lj1 == lj2 else 0

    da1, da2 = pathStrokeDashArray(path1), pathStrokeDashArray(path2)
    if da1 is None and da2 is None : 
        daScore = 1
    elif (da1 is None) != (da2 is None) : 
        daScore = 0 
    else : 
        da1 = parseDashArray(da1)
        da2 = parseDashArray(da2)
        numerator = sum(map(lambda a, b : a == b, da1, da2))
        denominator = max(len(da1), len(da2))
        daScore = numerator / denominator

    return avg([colorScore, lcScore, ljScore, daScore, swScore])

def autogroupColorSimilarity (doc, i, j, 
        containmentGraph=None) : 
    l1, a1, b1 = colorHistogram(doc, i, containmentGraph)
    l2, a2, b2 = colorHistogram(doc, j, containmentGraph)
    lscore = histScore(l1, l2)
    ascore = histScore(a1, a2)
    bscore = histScore(b1, b2)
    return avg([lscore, ascore, bscore])

def occludes (doc, i, j, **kwargs) : 
    """ does j occlude i? """ 
    areaIntersection = areaIntersectionDistance(doc, i, j)
    docArea = getDocBBox(doc).area()
    return j > i and (areaIntersection / docArea) > 1e-4

def bboxContains (doc, i, j, **kwargs) : 
    path1, path2 = _getPathPair(doc, i, j)
    b1 = pathBBox(path1.path)
    b2 = pathBBox(path2.path)
    return b1.contains(b2)

def relationshipGraph (doc, relFn, symmetric, **kwargs) : 
    """
    Encode relationship weight as edges. symmetric is either True or False.
    """
    paths = cachedPaths(doc)
    wtName = relFn.__name__
    n = len(paths)
    if symmetric : 
        G = nx.Graph()
        iterator = itertools.combinations
    else : 
        G = nx.DiGraph()
        iterator = itertools.permutations
    for i, j in iterator(range(n), r=2) : 
        wt = relFn(doc, i, j, **kwargs)
        G.add_edge(i, j, **{wtName: wt})
    # add self relation
    for i in range(n) : 
        wt = relFn(doc, i, i, **kwargs) 
        G.add_edge(i, i, **{wtName: wt})
    return G
