from functools import lru_cache, partial
from more_itertools import *
from itertools import *
import numpy as np
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import *
from vectorrvnn.utils import *
from vectorrvnn.geometry import *

AVG_VEL = 0.2
SMOOTHING = 0.90
END_PT_FACTOR = 0.9

def pointPathIntersect (path, point, radius):  
    p = Point(point['x'], point['y'])
    return enclosingGeometry(path).distance(p) < radius

def strokeCoversPath(path, stroke, radius): 
    fn = lambda pt : pointPathIntersect(path, pt, radius)
    return any(map(fn, stroke))

def samplePointsFromPolygon (polygon, k=20) :
    xm, ym, xM, yM = polygon.bounds
    samples = []
    while len(samples) < k : 
        x = rng.uniform(xm, xM)
        y = rng.uniform(ym, yM)
        if polygon.distance(Point(x, y)) < 1e-3 : 
            samples.append([x, y])
    return samples

def path2Polygon (path, k=20) : 
    pts = [path.point(t) for t in np.linspace(0, 1, k)]
    pts = [[p.real, p.imag] for p in pts]
    polygon = Polygon(pts)
    return polygon

def samplePointsFromPath (path, k=20) : 
    if path.isclosed() : 
        return samplePointsFromPolygon(path2Polygon(path, k), k)
    else :
        pts = [path.point(t) for t in np.linspace(0, 1, k)]
        pts = [[p.real, p.imag] for p in pts]
        return pts

def nodeScore (pathSet, paths, stroke, V, radius) :
    global END_PT_FACTOR
    relevantPaths = [paths[i].path for i in pathSet]
    startPt, endPt = stroke[0], stroke[-1]
    endsCheckFn = lambda x, y : pointPathIntersect(x, y, radius)
    hasEndPoint = any(starmap(
        endsCheckFn, 
        product([paths[i] for i in pathSet], [startPt, endPt])
    ))
    samples = map(samplePointsFromPath, relevantPaths)
    Pc = np.array(list(flatten(samples)))
    Ps = np.array([[s['x'], s['y']] for s in stroke])   
    cPoints = Pc.shape[0]
    M = squareform(pdist(np.concatenate((Pc, Ps))))
    M = M[:cPoints, cPoints:]
    Tc = M.min(axis=0)
    Ts = M.min(axis=1)
    muC, muS = np.mean(Tc), np.mean(Ts)
    stdC, stdS = np.std(Tc), np.std(Ts)
    beta = min(1, np.exp(1 - V) / 4)
    score = beta * (muC + muS) + (1 - beta) * (stdC + stdS)
    if hasEndPoint:  
        return score * END_PT_FACTOR
    return score

def strokeVelocity (stroke, doc) : 
    """ 
    Since the stroke points are sampled at equal 
    intervals, the stroke speed is simply the distance
    between strokes.
    """
    global AVG_VEL, SMOOTHING
    docbox = getDocBBox(doc)
    dim = max(docbox.h, docbox.w)
    stroke = aggregateDict(stroke, np.array)
    stroke = np.stack((stroke['x'], stroke['y'])).T / dim
    vels = np.sqrt(((stroke[1:] - stroke[:1]) ** 2).sum(1))
    vel = vels.mean()
    AVG_VEL = SMOOTHING * AVG_VEL + (1 - SMOOTHING) * vel
    return (vel - AVG_VEL)

def suggest (t, stroke, treeInferenceMethod, radius) : 
    paths = cachedPaths(t.doc)
    relevantPaths = [strokeCoversPath(p, stroke, radius) for p in paths]
    relevantPathIndices = [i for i, _ in enumerate(paths) if relevantPaths[i]]
    if len(relevantPathIndices) == 0 : return []
    t_ = treeInferenceMethod(t, subtrees=relevantPathIndices) 
    pathSets = [leavesInSubtree(t_, n) for n in t_.nodes]
    V = strokeVelocity(stroke, t.doc)
    pathSets.sort(key=partial(nodeScore, 
        paths=paths, stroke=stroke, V=V, radius=radius))
    pathSets = list(map(list, pathSets))
    bestThree = pathSets[:3]
    return bestThree
