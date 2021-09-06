""" vector descriptor for path geometries """
from skimage import color
import numpy as np
from functools import lru_cache
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

@lru_cache(maxsize=128)
def equiDistantSamples (doc, i, nSamples=5, **kwargs) :
    """ Sample points and concatenate to form a descriptor  """
    path = cachedPaths(doc)[i].path
    ts = np.linspace(0, 1, nSamples)
    L = path.length()
    pts = [path.point(path.ilength(t * L, 1e-1)) for t in ts]
    if kwargs['normalize'] : 
        dx, dy = docbb[2] - docbb[0], docbb[3] - docbb[1]
        x = [p.real / dx for p in pts]
        y = [p.imag / dy for p in pts]
        return [x,y]
    else : 
        x = [p.real for p in pts]
        y = [p.imag for p in pts]
        return [x,y]

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
