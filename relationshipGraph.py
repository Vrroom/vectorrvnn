import numpy as np
import copy
import itertools
import networkx as nx
from graphOps import subgraph
import matplotlib.image as image
from raster import rasterize, singlePathSvg, randomString

def optimalRotationAndTranslation (pts1, pts2) : 
    """
    Get the optimal rotation matrix
    and translation vector to transform
    pts1 to pts2 in linear least squares
    sense.

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
    pts1_ = pts1 - centroid1
    pts2_ = pts2 - centroid2
    pts1_ = pts1_ / np.std(pts1_, axis=0)
    pts2_ = pts2_ / np.std(pts2_, axis=0)
    H = pts1_.T @ pts2_
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    t = -(R @ centroid1.T) + centroid2.T
    return R, t, np.linalg.norm(pts2 - (pts1 @ R.T + t))

def isometryWithAlignment(path1, 
                          path2, 
                          ts1=np.arange(0, 1, 0.05),
                          ts2=np.arange(0, 1, 0.05)) :
    """
    Given two paths and an alignment, find
    the best isometric transformation which will
    take path1 to path2. 

    By definition of isometric transformation,
    we consider reflection, rotation and 
    translation.

    Return whether reflection was needed,
    the rotation matrix, the translation
    vector and the error.

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
    Find the best isometry to transform
    path1 into path2. 

    If the two paths are actually the
    same, then there are multiple 
    possible ways of matching them up. 
    
    Fortunately, because we know the 
    curve parametrizations, there are only
    a few ways of matching them up. 

    We loop over all possible matchings
    and find the transformation which
    minimizes the error.

    >>> paths = svg.Document('file.svg').flatten_all_paths()
    >>> ref, R, t, e = isometry(paths[0][0], paths[1][0])
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
        
def symGraph (paths, **kwargs) : 
    """
    Return a complete graph over the 
    paths. The edge attributes between 
    the vertices are the optimal
    transformations along with the 
    associated errors.

    Parameters
    ----------
    paths : list
        List of paths over which 
        symmetry has to be checked.
    """
    paths = list(enumerate(paths))
    G = nx.Graph() 
    for p1, p2 in itertools.product(paths, paths) :
        i, j = p1[0], p2[0]
        pi, pj = p1[1][0], p2[1][0]
        if i < j :
            r, R, t, err = isometry(pi, pj)
            G.add_edge(
                i, j, 
                reflection=r, 
                rotation=R.tolist(), 
                translation=t.tolist(),
                error=err,
                etype='symmetry'
            )
    return subgraph(G, lambda x : x['error'] < 1)

def adjGraph (paths, **kwargs) :
    """
    Return a graph over the 
    paths. We have coarse estimates
    of adjacency between two paths.
    
    We simply check whether their 
    bounding boxes intersect or not.

    Parameters
    ----------
    paths : list
        List of paths over which 
        adjacency has to be checked.
    """
    paths = list(enumerate(paths))
    G = nx.Graph()
    for i in range(len(paths)) :
        G.add_node(i)
    for p1, p2 in itertools.product(paths, paths) : 
        i, j = p1[0], p2[0]
        pi, pj = p1[1][0], p2[1][0]
        if i != j and pi != pj :
            if pi.approx_adj(pj) :
                G.add_edge(i, j, etype='adjacency')
    return G

def areaGraph (paths, **kwargs) : 
    """
    Return a complete graph over the 
    paths. Where the edge weights are the 
    areas of intersection between pairs
    of paths.
    
    Parameters
    ----------
    paths : list
        List of paths over which 
        adjacency has to be checked.
    """
    paths = list(enumerate(paths))
    G = nx.Graph()
    imageDict = dict() 
    for i, p in paths : 
        G.add_node(i)
        p_ = copy.deepcopy(p)
        p_[1].attrib = {'fill' : [0, 0, 0]}
        rs = randomString(10)
        singlePathSvg(p_, kwargs['vbox'], f'/tmp/{rs}.svg')
        rasterize(f'/tmp/{rs}.svg', f'/tmp/{rs}.png', makeSquare=False)
        img1 = image.imread(f'/tmp/{rs}.png')
        img1 = img1[:, :, 3]
        imageDict[i] = img1
    for p1, p2 in itertools.product(paths, paths) : 
        i, j = p1[0], p2[0]
        pi, pj = p1[1][0], p2[1][0]
        if i != j and pi != pj :
            intersect = imageDict[i] * imageDict[j]
            n, m = intersect.shape
            weight = intersect.sum() / (n * m) 
            G.add_edge(i, j, etype='area', weight=weight)
    return G

def pathIntersectionArea (path1, path2, vbox) : 
    """
    Calculate the area of intersection of two
    paths normalized by the document view box.

    This is done by first rasterizing and then
    taking the intersection of the bitmaps. 

    Parameters
    ----------
    path1 : svg.FlattenedPath
        Path 1
    path2 : svg.FlattenedPath
        Path 2
    vbox : list
        Viewbox of the document
    """
    # Fills with black hopefully
    fillDict = {'fill' : [0, 0, 0]}
    path1_ = copy.deepcopy(path1)
    path2_ = copy.deepcopy(path2)
    path1_[1].attrib = fillDict
    path2_[1].attrib = fillDict
    singlePathSvg(path1_, vbox, '/tmp/o1.svg')
    singlePathSvg(path2_, vbox, '/tmp/o2.svg')
    rasterize('/tmp/o1.svg', '/tmp/o1.png', makeSquare=False)
    rasterize('/tmp/o2.svg', '/tmp/o2.png', makeSquare=False)
    img1 = image.imread('/tmp/o1.png')
    img2 = image.imread('/tmp/o2.png')
    img1 = img1[:, :, 3]
    img2 = img2[:, :, 3]
    intersect = img1 * img2
    n, m = intersect.shape
    fraction = intersect.sum() / (n * m)
    return fraction
