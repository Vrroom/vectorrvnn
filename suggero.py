from complexOps import *
from descriptor import *
from TripletDataset import *
from treeCompare import ted
from treeOps import *
import math
from skimage import color
from matplotlib import colors
import numpy as np
from itertools import product
from sklearn.cluster import AgglomerativeClustering
import svgpathtools as svg
from copy import deepcopy
from collections import namedtuple

def hac2nxDiGraph (leaves, links) :
    T = nx.DiGraph()
    T.add_nodes_from(leaves)
    n = len(leaves) 
    for i, link in enumerate(links) : 
        T.add_edge(n + i, link[0])
        T.add_edge(n + i, link[1])
    for node in T.nodes:  
        T.nodes[node]['pathSet'] = list(leavesInSubtree(T, node))
    return T

def avgDistancePoint2Path (point, path) : 
    line = equiDistantSamples(path.path, nSamples=10, normalize=False)
    line = np.array(line).T
    return np.linalg.norm(line - point, axis=1).mean()

def pathsWithTolerance (paths) : 
    newPaths = []
    for path in paths:  
        FlattenedPath = namedtuple('FlattenedPath', (*path._fields, 'tol'))
        start, end = path.path.start, path.path.end
        start = np.array([start.real, start.imag])
        end = np.array([end.real, end.imag])
        tol1 = min(avgDistancePoint2Path(start, p) for p in paths)
        tol2 = min(avgDistancePoint2Path(end, p) for p in paths)
        tol = min(tol1, tol2)
        newPaths.append(FlattenedPath(*path, tol=tol))
    return newPaths

def affinityMatrix (paths, affinityFn) :
    n = len(paths)
    M = np.zeros((n, n))
    for i, j in product(range(n), range(n)) :
        M[i, j] = affinityFn(paths[i], paths[j])
    M = M / (M.max() + 1e-3)
    return M 

def circlesIntersectionArea(circle1, circle2) : 
    r, center1 = circle1
    R, center2 = circle2
    d = abs(center1 - center2) 
    eps = 1e-3
    v1 = np.clip((d ** 2 + r ** 2 - R ** 2)/(2 * d * r + 1e-3), -1, 1)
    t1 = r ** 2 * np.arccos(v1)
    v2 = np.clip((d ** 2 + R ** 2 - r ** 2)/(2 * d * R + 1e-3), -1, 1)
    t2 = R ** 2 * np.arccos(v2)
    t3 = 0.5 * np.sqrt(max(0, (-d + r + R) * (-d - r + R) * (-d + r + R) * (d + r + R)))
    return t1 + t2 - t3

def circleArea (circle) : 
    r, _ = circle
    return np.pi * (r ** 2)

def circleIoU (circle1, circle2) : 
    intersectionArea = circlesIntersectionArea(circle1, circle2)
    unionArea = circleArea(circle1) + circleArea(circle2) - intersectionArea
    return intersectionArea / unionArea

def center (bbox) : 
    xm, xM, ym, yM = bbox
    return complex((xm + xM) / 2, (ym + yM) / 2)

def getPathColorAttribute (path, attr) : 
    if attr in path.element.attrib : 
        attr = colors.to_rgb(path.element.attrib[attr])
    else :
        attr = [1, 1, 1]
    return attr

def getStrokeAttribute (path) :
    if 'stroke-width' in path.element.attrib : 
        return float(path.element.attrib['stroke-width'])
    else :
        return 1

def consecutiveAngles (line)  :
    angles = []
    for x, y, z in zip(line, line[1:], line[2:]): 
        v1 = y - x
        v2 = z - y
        cosTheta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosTheta = np.clip(cosTheta, -1, 1)
        angles.append(np.arccos(cosTheta))
    return np.array(angles)

#####################################
## PROXIMITY AFFINITY
#####################################

# Local Proximity between paths
# = distance between center of bounding boxes of
#   the paths.
# def localProximityAffinity(path1, path2) : 
#     center1 = center(path1.path.bbox())
#     center2 = center(path2.path.bbox())
#     sqDist = complexDot(center1 - center2, center1 - center2)
#     return math.sqrt(sqDist)
# 
# def globalProximityAffinity(path1, path2) :
#     samples1 = equiDistantSamples(path1.path, nSamples=10, normalize=False)
#     samples2 = equiDistantSamples(path2.path, nSamples=10, normalize=False)
#     samples1 = np.array(samples1)
#     samples2 = np.array(samples2)
#     return np.linalg.norm(samples1 - samples2, axis=0).mean()

#####################################
## ENDPOINT CONNECTIVITY AFFINITY
#####################################

# def endpointAffinity (path1, path2) :
#     r1 = path1.tol
#     r2 = path2.tol
#     c1 = (r1, path1.path.start)
#     c2 = (r1, path1.path.end)
#     c3 = (r2, path2.path.start)
#     c4 = (r2, path2.path.end)
#     score = 1 - max(circleIoU(c1, c3), circleIoU(c1, c4), circleIoU(c2, c3), circleIoU(c2, c4))
#     return score

#####################################
## PARALLELISM AFFINITY
#####################################

# def angleAffinity (path1, path2): 
#     line1 = equiDistantSamples(path1.path, nSamples=10, normalize=False)
#     line2 = equiDistantSamples(path2.path, nSamples=10, normalize=False)
#     line1 = np.array(line1).T
#     line2 = np.array(line2).T
#     angles1 = consecutiveAngles(line1)
#     angles2 = consecutiveAngles(line2)
#     return np.linalg.norm(angles1 - angles2)

#####################################
## SIMILARITY AFFINITY
#####################################

# def fourierDescriptorAffinity (path1, path2) : 
#     return np.linalg.norm(fd(path1.path) - fd(path2.path))
# 
def fillAffinity (path1, path2) : 
    lab1 = color.rgb2lab(getPathColorAttribute(path1, 'fill'))
    lab2 = color.rgb2lab(getPathColorAttribute(path2, 'fill'))
    return np.linalg.norm(lab1 - lab2)

def strokeAffinity (path1, path2) : 
    lab1 = color.rgb2lab(getPathColorAttribute(path1, 'stroke'))
    lab2 = color.rgb2lab(getPathColorAttribute(path2, 'stroke'))
    return np.linalg.norm(lab1 - lab2)

def strokeWidthAffinity (path1, path2) : 
    return abs(getStrokeAttribute(path1) - getStrokeAttribute(path2))

#####################################
## EVALUATE SUGGERO
#####################################

def combinedAffinityMatrix (paths, affinityFns, weights) : 
    affinityMatrices = [affinityMatrix(paths, fn) for fn in affinityFns]
    combinedMatrix = sum(w * M for w, M in zip(weights, affinityMatrices))
    return combinedMatrix

def suggero (t) : 
    doc = svg.Document(t.svgFile)
    paths = doc.flatten_all_paths()
    paths = pathsWithTolerance(paths)
    fnKeys = filter(lambda x : x.endswith('Affinity'), globals().keys())
    affinityFns = [globals()[fnName] for fnName in fnKeys]
    nfns = len(affinityFns)
    uniformWts = (1 / nfns) * np.ones(nfns)
    M = combinedAffinityMatrix(paths, affinityFns, uniformWts)
    agg = AgglomerativeClustering(1, affinity='precomputed', linkage='single')
    agg.fit(M)
    return hac2nxDiGraph(list(range(len(paths))), agg.children_)

if __name__ == "__main__" : 
    testData = TripletSVGDataSet('cv64.pkl').svgDatas[:1]
    inferredTrees = [suggero(t) for t in testData]
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    print(scores)
