""" geometry utility functions """
import numpy as np

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
