import numpy as np
import copy
import itertools
import networkx as nx
from graphOps import subgraph
import matplotlib.image as image
from raster import rasterize, singlePathSvg, randomString

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
