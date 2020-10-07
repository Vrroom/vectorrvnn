import numpy as np
import random
from complexOps import *

def d2 (path, docbb, bins=10, nSamples=100, **kwargs) :
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

def shell(path, docbb, bins=10, **kwargs) :
    """
    Like d2 but instead of random points,
    we use points at fixed intervals on
    the curve.

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
    xmin, xmax, ymin, ymax = path.bbox()
    rmax = np.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)
    K = 100
    L = path.length()
    samples = []
    for i in range(K) : 
        s = (i / K) * L
        samples.append(path.point(path.ilength(s, s_tol=1e-6)))
    d2 = [abs(x-y) for x in samples for y in samples if complexLE(x, y)]
    hist = np.zeros(bins)
    binSz = rmax / bins
    if binSz == 0 : 
        return np.ones(bins)
    for dist in d2 :
        b = int(dist/binSz)
        if b >= bins : 
            b = bins - 1
        hist[b] += 1
    return hist

def a3 (path, docbb, bins=10, nSamples=100, **kwargs) :
    """
    Compute the a3 descriptors of the path.
    Take three random points on the curve
    and make an histogram of the angle 
    subtended.

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
    hist = np.zeros(bins)
    binSz = np.pi / bins
    L = path.length()
    i = 0
    while i < nSamples :

        pt1 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt2 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt3 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        
        v1 = pt1 - pt3
        v2 = pt2 - pt3
        
        if v1 == 0 or v2 == 0 : 
            continue
        
        cos = complexDot(v1, v2) / (abs(v1) * abs(v2))
        cos = np.clip(cos, -1, 1)

        theta = np.arccos(cos)

        bIdx = int(theta / binSz)

        if bIdx == bins : 
            bIdx -= 1

        hist[bIdx] += 1

        i += 1

    return hist

def d3 (path, docbb, bins=10, nSamples=100, **kwargs) :
    """
    Compute the d3 descriptors of the path.
    Take three random points on the curve
    and make their area histogram.

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
    xmin, xmax, ymin, ymax = path.bbox()
    area = (ymax - ymin) * (xmax - xmin) 
    if area <= 1e-10 : 
        return np.ones(bins)
    hist = np.zeros(bins)
    binSz = area / bins
    L = path.length() 
    for i in range(nSamples) : 
        pt1 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt2 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        pt3 = path.point(path.ilength(random.random() * L, s_tol=1e-3))
        v1 = pt1 - pt3
        v2 = pt2 - pt3
        trArea = complexCross(v1, v2) / 2
        bIdx = int(trArea / binSz)
        if bIdx == bins :
            bIdx -= 1
        hist[bIdx] += 1
    return hist 

def fd (path, docbb, nSamples=100, freqs=10, **kwargs) :
    """
    Compute the fourier descriptors of the
    path with respect to its centroid.

    Parameters
    ----------
    path : svg.Path
        Input path. 
    docbb : list
        Bounding Box of the document.
    nSamples : int
        Sampling frequency for the path
    """
    ts = np.arange(0, 1, 1 / nSamples)
    L = path.length()
    if L == 0 :
        return np.ones(min(nSamples, 20))
    pts = np.array([path.point(path.ilength(t * L, s_tol=1e-5)) for t in ts])
    pts = pts - pts.mean()
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

def relbb (path, docbb, **kwargs) :
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
    xmin, xmax, ymin, ymax = path.bbox()
    x1 = (xmin - docbb[0]) / (docbb[2] - docbb[0])
    x2 = (xmax - docbb[0]) / (docbb[2] - docbb[0])
    y1 = (ymin - docbb[1]) / (docbb[3] - docbb[1])
    y2 = (ymax - docbb[1]) / (docbb[3] - docbb[1])
    return [x1, y1, x2 - x1, y2 - y1]

def equiDistantSamples (path, docbb, nSamples=5, **kwargs) :
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
    ts = np.linspace(0, 1, nSamples)
    L = path.length()
    pts = [path.point(path.ilength(t * L, 1e-4)) for t in ts]
    dx, dy = docbb[2] - docbb[0], docbb[3] - docbb[1]
    x = [p.real / dx for p in pts]
    y = [p.imag / dy for p in pts]
    return [x,y]

def oneHot (path, docbb, **kwargs) :
    index = kwargs['index']
    nPaths = kwargs['nPaths']
    desc = [0] * nPaths
    desc[index] = 1
    return desc

