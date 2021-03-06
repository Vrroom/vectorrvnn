""" geometry utility functions """
import numpy as np
from vectorrvnn.utils.svg import cachedPaths
from skimage import color
from shapely.geometry import *
from scipy.spatial import ConvexHull
import svgpathtools as svg
from copy import deepcopy

def histScore (a, b) : 
    return np.minimum(a, b).sum() / (np.maximum(a, b).sum() + 1e-5)

def normalizedCiede2000Score (rgb1, rgb2) : 
    lab1 = color.rgb2lab(rgb1)
    lab2 = color.rgb2lab(rgb2)
    cmin = (0, -128, -128)
    cmax = (100, 127, 127)
    maxD = color.deltaE_ciede2000(cmin, cmax)
    delE = color.deltaE_ciede2000(lab1, lab2)
    return 1 - (delE / maxD)

def distanceOfPointFromLine (p, line) : 
    """ exactly what the name suggests :/ """
    p_, slope = line
    A, B = -slope[1], slope[0]
    C = -A * p_[0] - B * p_[1]
    return abs(A * p[0] + B * p[1] + C) / (np.sqrt(A ** 2 + B ** 2) + 1e-3)

def avgDistancePoint2Path (point, doc, i) : 
    """ avg distance of a point to the ith path in doc """
    path = cachedPaths(doc)[i].path
    ts = np.linspace(0, 1, 10)
    L = path.length()
    line = [path.point(path.ilength(t * L, 1e-4)) for t in ts]
    x = [p.real for p in line]
    y = [p.imag for p in line]
    line = [x, y]
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

def optimalRotationAndTranslation (pts1, pts2) : 
    """
    Get the optimal rotation matrix
    and translation vector to transform
    pts1 to pts2 in linear least squares
    sense.

        pts2 = R * pts1 + t

    The output is the rotation matrix,
    the translation vector and the error.

    Parameters
    ----------
    pts1 : np.ndarray
        Points in the first set
    pts2 : np.ndarray
        Points in the second set
    """
    centroid1 = np.mean(pts1, axis=0)
    centroid2 = np.mean(pts2, axis=0)
    # center both point sets
    pts1_ = pts1 - centroid1
    pts2_ = pts2 - centroid2
    # scale parameter
    std = np.stack((pts1_, pts2_)).ravel().std()
    # normalize by scale parameter
    pts1_ = pts1_ / (std + 1e-5)
    pts2_ = pts2_ / (std + 1e-5)
    H = pts1_.T @ pts2_
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # translation vector
    t = centroid2.T - (R @ centroid1).T
    e = np.linalg.norm(pts2 - (pts1 @ R.T + t))
    relE = e / (np.linalg.norm(pts1) + np.linalg.norm(pts2))
    return R, t, relE

def isometryWithAlignment(path1, path2, 
        ts1=np.arange(0, 1, 0.05),
        ts2=np.arange(0, 1, 0.05)) :
    """
    Given two paths and an alignment, find the best isometric 
    transformation which will take path1 to path2. 

    Return whether reflection was needed, the rotation matrix, 
    the translation vector and the relative error.

    Parameters
    ----------
    path1 : svg.Path
        The first path.
    path2 : svg.Path
        The second path.
    ts1 : np.ndarray
        The sequence of curve parametrization
        points for path1.
    ts2 : np.ndarray
        The sequence of curve parametrization
        points for path2. The correspondence
        between points on each path
        uses these two sequences.
    """

    reflection = np.array([[1, 0], [0, -1]]) 
    pts1 = []
    pts2 = []
    for t1, t2 in zip(ts1, ts2) : 
        pt1 = path1.point(t1)
        pt2 = path2.point(t2)
        pts1.append([pt1.real, pt1.imag])
        pts2.append([pt2.real, pt2.imag])
    pts1 = np.array(pts1)
    pts2 = np.array(pts2)
    refPts1 = pts1 @ reflection.T
    R1, t1, e1 = optimalRotationAndTranslation(pts1, pts2)
    R2, t2, e2 = optimalRotationAndTranslation(refPts1, pts2)
    if e1 < e2 : 
        return False, R1, t1, e1
    else :
        return True, R2, t2, e2 

def isometry(path1, path2) :
    """
    Find the best isometry to transform path1 into path2. 

    Because we know the curve parametrizations, there are 
    only a few ways of matching them up. 

    We loop over all possible matchings and find the 
    transformation which minimizes the error.

    >>> paths = svg.Document('file.svg').paths()
    >>> ref, R, t, e = isometry(paths[0].path, paths[1].path)
    >>> print("Reflection: ", ref)
    >>> print("Rotation: ", R)
    >>> print("Translation: ", t)
    >>> print("Error: ", e)

    Parameters
    ----------
    path1 : svg.Path
        The first path.
    path2 : svg.Path
        The second path.
    """
    l1 = path1.length()
    l2 = path2.length()
    if path1.isclosed() and path2.isclosed() : 
        r, R, t, err = None, None, None, None
        rng = np.arange(0, 1, 0.05)
        for i in range(len(rng))  :
            fst = rng[i:]
            snd = rng[:i]
            ts = np.concatenate((fst, snd))
            r_, R_, t_, err_ = isometryWithAlignment(path1, path2, ts1=ts)
            if not err :
                r, R, t, err = r_, R_, t_, err_
            elif err_ < err :
                r, R, t, err = r_, R_, t_, err_
        return r, R, t, err
    else :
        return isometryWithAlignment(path1, path2)

def enclosingGeometry(path) : 
    path_ = deepcopy(path)
    path_.path.approximate_arcs_with_cubics()
    pts = []
    for seg in path_.path :
        if isinstance(seg, svg.Line) :
            pts.append(seg.start)
            pts.append(seg.end)
        elif isinstance(seg, svg.QuadraticBezier) :
            pts.append(seg.start)
            pts.append(seg.control)
            pts.append(seg.end)
        elif isinstance(seg, svg.CubicBezier) :
            pts.append(seg.start)
            pts.append(seg.control1)
            pts.append(seg.control2)
            pts.append(seg.end)
    pts = np.array([[pt.real, pt.imag] for pt in pts])
    if len(pts) >= 3 :
        return Polygon(pts[ConvexHull(pts).vertices])
    elif len(pts) == 2 :
        return LineString(pts.tolist())
    else : 
        return Point(pts.tolist().pop())

def flattenCubic(curve, tol=1e-2) : 
    def flat(c, tol) : 
        u = 3 * c.control1 - 2 * c.start - c.end
        v = 3 * c.control2 - 2 * c.end - c.start
        w = complex(max(u.real, v.real), max(u.imag, v.imag))
        return abs(w) < tol
    if flat(curve, tol): 
        return [svg.Line(curve.start, curve.end)]
    else : 
        l, r = list(map(
            lambda x : svg.CubicBezier(*x), 
            svg.split_bezier(curve, 0.5)
        ))
        return flattenCubic(l, tol) + flattenCubic(r, tol)

def quad2cubic (quad) : 
    s, c, e = quad.bpoints() 
    c1 = s + (2 / 3) * (c - s)
    c2 = e + (2 / 3) * (c - e) 
    return svg.CubicBezier(s, c1, c2, e)

def flattenPath (path, tol=1e-2) :
    lines = []
    for curve in path :
        if isinstance(curve, svg.Line) : 
            lines.append(curve)
        elif isinstance(curve, svg.CubicBezier) :
            lines.extend(flattenCubic(curve, tol))
        elif isinstance(curve, svg.Arc) : 
            cubicApprox = curve.as_cubic_curves(4) 
            for cubic in cubicApprox : 
                lines.extend(flattenCubic(cubic, tol))
        elif isinstance(curve, svg.QuadraticBezier): 
            lines.extend(flattenCubic(quad2cubic(curve), tol))
    return lines

