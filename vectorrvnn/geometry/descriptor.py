""" vector descriptor for path geometries """
from skimage import color
import numpy as np
from functools import lru_cache
from itertools import starmap
from bisect import bisect
from copy import deepcopy
from vectorrvnn.utils.comp import *
from vectorrvnn.utils.svg import *
from vectorrvnn.utils.graph import *
from vectorrvnn.utils.boxes import *
from vectorrvnn.utils.random import *

@lru_cache(maxsize=128)
def colorHistogram(doc, i, containmentGraph=None, threadLocal=False) : 
    """ channel wise histograms in lab space """
    if containmentGraph is None : 
        subset = subsetSvg(doc, [i])
    else : 
        subset = subsetSvg(doc, 
                list(descendants(containmentGraph, i)))
    box = getDocBBox(subset)
    if box.w < box.h : 
        w, h = 200, (box.h / box.w) * 200
    else : 
        w, h = (box.w / box.h) * 200, 200
    im = rasterize(subset, w=int(w), h=int(h), threadLocal=threadLocal)
    rgb, alpha = im[:, :, :3], im[:, :, 3]
    lab = color.rgb2lab(rgb) 
    l = lab[:, :, 0][alpha > 0]
    a = lab[:, :, 1][alpha > 0]
    b = lab[:, :, 2][alpha > 0]
    histL = np.histogram(l, range=(0, 100)   , bins=100)[0]
    histA = np.histogram(a, range=(-128, 127), bins=255)[0]
    histB = np.histogram(b, range=(-128, 127), bins=255)[0]
    return histL, histA, histB

@lru_cache(maxsize=128)
def d2 (doc, i, bins=10, nSamples=100, **kwargs) :
    """
    Compute the d2 descriptors of the path.
    Take two random points on the curve
    and make an histogram of the distance 
    between them. 
    """
    path = cachedPaths(doc)[i].path
    if pathBBox(path).isDegenerate() : 
        return np.ones(bins)
    L = path.length() 
    rs = []
    for i in range(nSamples) : 
        pt1 = path.point(path.ilength(rng.random() * L, s_tol=1e-2))
        pt2 = path.point(path.ilength(rng.random() * L, s_tol=1e-2))
        rs.append(abs(pt1 - pt2))
    return np.histogram(rs, bins=bins)[0] / nSamples

@lru_cache(maxsize=128)
def shapeHistogram (doc, i, nSamples=50) : 
    path = cachedPaths(doc)[i].path
    box = pathBBox(path)
    s = max(box.w, box.h) * np.sqrt(2)
    ts = np.arange(0, 1, 1 / nSamples)
    L = path.length()
    if L == 0 :
        return np.ones(nSamples)
    pts = np.array([path.point(path.ilength(t * L, s_tol=1e-2)) for t in ts])
    pts = pts - pts.mean()
    pts = np.abs(pts) / s
    return np.histogram(pts, range=(0, 1), bins=nSamples)[0]

def shapeHistogramPolyline (doc, i, nSamples=50, lines=None) : 
    assert lines is not None, "Incorrect arguments" 
    box = pathBBox(svg.Path(*lines[i]))
    s = max(box.w, box.h) * np.sqrt(2)
    pts = np.array(equiDistantPointsOnPolyline(doc, lines[i], nSamples=nSamples)).T
    pts = pts - pts.mean(0)
    pts = np.linalg.norm(pts, axis=1) / s
    return np.histogram(pts, range=(0, 1), bins=nSamples)[0]

@lru_cache(maxsize=128)
def fd (doc, i, nSamples=25, freqs=10, **kwargs) :
    """
    Compute the fourier descriptors of the
    path with respect to its centroid.
    """
    path = cachedPaths(doc)[i].path
    ts = np.arange(0, 1, 1 / nSamples)
    L = path.length()
    if L == 0 :
        return np.ones(min(nSamples, 20))
    pts = np.array([path.point(path.ilength(t * L, s_tol=1e-2)) for t in ts])
    pts = pts - pts.mean()
    pts = np.abs(pts)
    an = np.fft.fft(pts)
    pos = an[1:nSamples//2]
    neg = an[nSamples//2 + 1:]
    pos = pos[:freqs]
    neg = neg[-freqs:]
    newAn = np.hstack([an[0], pos, neg])
    reals = np.array(list(map(lambda x : x.real, newAn)))
    imags = np.array(list(map(lambda x : x.imag, newAn)))
    newAn = np.hstack([reals, imags])
    return newAn

@lru_cache(maxsize=128)
def bb (doc, i, **kwargs) : 
    """ absolute bounding box coordinates for the ith path """
    path = cachedPaths(doc)[i].path
    return pathBBox(path).tolist()

@lru_cache(maxsize=128)
def relbb (doc, i, **kwargs) :
    """ 
    Compute the relative bounding box of the path 
    with respect to the document's bounding box.
    """
    docbb = getDocBBox(doc)
    path = cachedPaths(doc)[i].path

    xmin, xmax, ymin, ymax = pathBBox(path).tolist()
    x1 = (xmin - docbb.x) / (docbb.w)
    x2 = (xmax - docbb.x) / (docbb.w)
    y1 = (ymin - docbb.y) / (docbb.h)
    y2 = (ymax - docbb.y) / (docbb.h)
    return [x1, y1, x2 - x1, y2 - y1]

def normalizePts2Doc (doc, pts) :
    """ Normalize points on a document to be in [-1, 1] by [-1, 1]

    pts is a list of complex points
    """
    docbox = getDocBBox(doc)
    x, y = docbox.x, docbox.y
    dx, dy = docbox.w, docbox.h
    # set points in [0, 1] by [0, 1]
    xs = [(p.real - x) / dx for p in pts]
    ys = [1 - ((p.imag - y) / dy) for p in pts]
    # transform them to [-1, 1] by [-1, 1]
    xs = [2 * (x - 0.5) for x in xs]
    ys = [2 * (y - 0.5) for y in ys]
    return list(starmap(complex, zip(xs, ys)))

def equiDistantPointsOnPolyline (doc, poly, nSamples=5, **kwargs) :
    """ Sample equidistant points on a polyline """
    eps = 1e-10
    ts = np.linspace(0, 1 - eps, nSamples)
    lens = [l.length() for l in poly]
    clens = deepcopy(lens)
    ss = sum(lens) * ts
    for i in range(1, len(clens)) :
        clens[i] += clens[i - 1]
    pts = []
    if sum(lens) == 0 :
        pts = [poly[0].start for _ in range(nSamples)] 
    else : 
        for s in ss :
            i = bisect(clens, s)
            diff = 0 if i == 0 else (s - clens[i - 1])
            pts.append(poly[i].point(diff / lens[i]))
    if kwargs.get('normalize', False) :
        pts = normalizePts2Doc(doc, pts)
    x = [p.real for p in pts]
    y = [p.imag for p in pts]
    return [x,y]
    
def samples (doc, path, nSamples=5, **kwargs) :
    """ Draw samples from a path. 
    
    Samples aren't equidistant but we avoid 
    expensive inverse arclength computation.
    """
    ts = np.linspace(0, 1, nSamples)
    pts = [path.point(t) for t in ts]
    if kwargs.get('normalize', False): 
        pts = normalizePts2Doc(doc, pts)
    x = [p.real for p in pts]
    y = [p.imag for p in pts]
    return [x,y]

def equiDistantSamples (doc, path, nSamples=5, **kwargs) :
    """ Sample points and concatenate to form a descriptor  """
    ts = np.linspace(0, 1, nSamples)
    L = path.length()
    pts = [path.point(path.ilength(t * L, 1e-1)) for t in ts]
    if kwargs.get('normalize', False): 
        pts = normalizePts2Doc(doc, pts)
    x = [p.real for p in pts]
    y = [p.imag for p in pts]
    return [x,y]

@lru_cache(maxsize=128)
def memoEquiDistantSamples (doc, i, nSamples=5, **kwargs) :
    path = cachedPaths(doc)[i].path
    return equiDistantSamples(doc, path, nSamples, **kwargs)

@lru_cache(maxsize=128)
def pathBitmap (doc, i, fill=True, threadLocal=False, **kwargs) : 
    doc_ = subsetSvg(doc, [i])
    if fill : 
        for path in doc_.paths() : 
            path.element.attrib.pop('style', None)
            path.element.attrib['fill'] = 'black'
    box = getDocBBox(doc_)
    if box.w < box.h : 
        w, h = 200, (box.h / box.w) * 200
    else : 
        w, h = (box.w / box.h) * 200, 200
    im = rasterize(doc_, w=int(w), h=int(h), threadLocal=threadLocal)
    return im[:, :, 3]
