""" geometry utility functions """
import numpy as np
from vectorrvnn.utils.svg import cachedPaths

def isDegenerateBBox (box) :
    _, _, h, w = box
    return h == 0 and w == 0

def bboxUnion(a, b) : 
    ax1, ax2, ay1, ay2 = a
    bx1, bx2, by1, by2 = b
    return [min(ax1, bx1), 
            max(ax2, bx2), 
            min(ay1, by1), 
            max(ay2, by2)]

def graphicBBox (doc) : 
    paths = cachedPaths(doc)
    bboxes = [p.path.bbox() for p in paths]
    x, X, y, Y = reduce(bboxUnion, bboxes)
    return [x, y, X - x, Y - y]

def distanceOfPointFromLine (p, line) : 
    """ exactly what the name suggests :/ """
    p_, slope = line
    A, B = -slope[1], slope[0]
    C = -A * p_[0] - B * p_[1]
    return abs(A * p[0] + B * p[1] + C) / (np.sqrt(A ** 2 + B ** 2) + 1e-3)

def avgDistancePoint2Path (point, doc, i) : 
    """ avg distance of a point to the ith path in doc """
    line = equiDistantSamples(doc, i, nSamples=10, normalize=False)
    line = np.array(line).T
    return np.linalg.norm(line - point, axis=1).mean()

def circlesIntersectionArea(circle1, circle2) : 
    """ 
    Find the area of intersection of two circles. 
    Probably stolen from wolfram alpha.
    """
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
    """ area of circle """
    r, _ = circle
    return np.pi * (r ** 2)

def center (bbox) : 
    """ center of a bounding box """
    xm, xM, ym, yM = bbox
    return complex((xm + xM) / 2, (ym + yM) / 2)
