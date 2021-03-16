import numpy as np
import random
from complexOps import *
import matplotlib.colors as colors
from functools import lru_cache
from strokeAnalyses import cachedFlattenPaths

def d2 (doc, i , docbb, bins=10, nSamples=100, **kwargs) :
    """
    Compute the d2 descriptors of the path.
    Take two random points on the curve
    and make an histogram of the distance 
    between them. We use this or fd in our
    experiments.

    Parameters
    ----------
    path : svg.Path
        Input path.
    docbb : list
        bounding box of the document.
    bins : int
        Number of histogram bins. Also the
        dimension of the descriptor.
    nSamples : int
        Number of samples while making
        histogram.
    """
    path = cachedFlattenPaths(doc)[i].path
    xmin, xmax, ymin, ymax = path.bbox()
    rmax = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)
    if rmax <= 1e-10 : 
        return np.ones(bins)
    hist = np.zeros(bins)
    binSz = rmax / bins
    L = path.length() 
    for i in range(nSamples) : 
        pt1 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt2 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        r = abs(pt1 - pt2) 
        bIdx = int(r / binSz)
        if bIdx == bins :
            bIdx -= 1
        hist[bIdx] += 1
    hist = hist / hist.sum()
    return hist.tolist()

@lru_cache(maxsize=128)
def fd (doc, i, nSamples=25, freqs=10, **kwargs) :
    """
    Compute the fourier descriptors of the
    path with respect to its centroid.

    Parameters
    ----------
    path : svg.Path
        Input path. 
    nSamples : int
        Sampling frequency for the path
    """
    path = cachedFlattenPaths(doc)[i].path
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

def bb (doc, i) : 
    path = cachedFlattenPaths(doc)[i].path
    x1, x2, y1, y2 = path.bbox()
    return [x1, y1, x2 - x1, y2 - y1]

def relbb (doc, i, **kwargs) :
    """ 
    Compute the relative bounding box
    of the path with respect to the 
    document's bounding box.

    Parameters
    ----------
    path : svg.Path
        Input path.
    docbb : list
        The svg document's bounding box.
    """
    docbb = doc.get_viewbox()
    path = cachedFlattenPaths(doc)[i].path
    xmin, xmax, ymin, ymax = path.bbox()
    x1 = (xmin - docbb[0]) / (docbb[2] - docbb[0])
    x2 = (xmax - docbb[0]) / (docbb[2] - docbb[0])
    y1 = (ymin - docbb[1]) / (docbb[3] - docbb[1])
    y2 = (ymax - docbb[1]) / (docbb[3] - docbb[1])
    return [x1, y1, x2 - x1, y2 - y1]

@lru_cache(maxsize=128)
def equiDistantSamples (doc, i, nSamples=5, **kwargs) :
    """
    Sample points and concatenate to form a descriptor.

    Parameters
    ----------
    path : svg.Path
        Input path. 
    docbb : list
        Bounding Box of the document.
    nSamples : int
        Sampling frequency for the path
    """
    path = cachedFlattenPaths(doc)[i].path
    ts = np.linspace(0, 1, nSamples)
    L = path.length()
    pts = [path.point(path.ilength(t * L, 1e-4)) for t in ts]
    if kwargs['normalize'] : 
        dx, dy = docbb[2] - docbb[0], docbb[3] - docbb[1]
        x = [p.real / dx for p in pts]
        y = [p.imag / dy for p in pts]
        return [x,y]
    else : 
        x = [p.real for p in pts]
        y = [p.imag for p in pts]
        return [x,y]
