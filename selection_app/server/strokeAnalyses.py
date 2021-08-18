from functools import lru_cache, partial
from more_itertools import flatten
import numpy as np
from scipy.spatial.distance import pdist, squareform
from shapely.geometry import *
from vectorrvnn.utils import *

def strokeCoversPath(path, stroke, doc, radius): 
    def pointPathIntersect (point):  
        r = (radius / 100) * max(vbox[2], vbox[3])
        p = Point(point['x'], point['y'])
        return path.path.hull.distance(p) < r
    vbox = doc.get_viewbox() 
    return any(map(pointPathIntersect, stroke))

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

def nodeScore (pathSet, paths, stroke) :
    relevantPaths = [paths[i].path for i in pathSet]
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
    beta = 0.5
    return beta * (muC + muS) + (1 - beta) * (stdC + stdS)

def suggest (t, stroke, treeInferenceMethod, radius) : 
    paths = cachedFlattenPaths(t.doc)
    relevantPaths = [strokeCoversPath(p, stroke, t.doc, radius) for p in paths]
    relevantPathIndices = [i for i, _ in enumerate(paths) if relevantPaths[i]]
    if len(relevantPathIndices) == 0 : 
        return []
    t_ = treeInferenceMethod(t, subtrees=relevantPathIndices) 
    pathSets = [leavesInSubtree(t_, n) for n in t_.nodes]
    pathSets.sort(key=partial(nodeScore, paths=paths, stroke=stroke))
    pathSets = list(map(list, pathSets))
    bestThree = pathSets[:3]
    return bestThree
