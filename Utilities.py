# Avoid repetition at all costs.
import os
import os.path as osp
import json
import string
import subprocess
import more_itertools
import itertools
import networkx as nx
import xml.etree.ElementTree as ET
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms import bipartite
from networkx.algorithms import community
import copy
import multiprocessing as mp
import svgpathtools as svg
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.image as image
from matplotlib.offsetbox import TextArea
from matplotlib.offsetbox import DrawingArea
from matplotlib.offsetbox import OffsetImage
from matplotlib.offsetbox import AnnotationBbox
from imageio import imwrite
import math
from skimage import transform

def argmax(l) :
    """
    Compute the argmax of a list.

    Parameters
    ----------
    l : list
    """
    return next(filter(lambda x : max(l) == b[x], range(len(l))))

def argmin(l) :
    """
    Compute the argmin of a list.

    Parameters
    ----------
    l : list
    """
    return next(filter(lambda x : min(l) == b[x], range(len(l))))

def zipDirs (dirList) :
    """ 
    A common operation is to get SVGs
    or graphs from different directories
    and match them up and perform 
    some operations on them. 

    Since the files have similar names,
    we sort them before zipping them.

    For example, dir1 may have E269.svg
    while dir2 may have E269.json. Sorting 
    will ensure that while zipping, these 
    two files are together.

    Parameters
    ----------
    dirList : list
        List of directories to zip.
    """
    filesInDirList = list(map(os.listdir, dirList))
    for i in range(len(dirList)) :
        filesInDirList[i].sort()
        filesInDirList[i] = [os.path.join(dirList[i], f) for f in filesInDirList[i]]
    return zip(*filesInDirList)


def subgraph (G, predicate=lambda x : x['weight'] < 1e-3) :
    """ 
    Get a subgraph of G by
    removing edges which don't 
    satisfy a given predicate.

    Useful when filtering those edges
    from the symmetry graph which have
    high error.

    Parameters
    ----------
    G : nx.Graph()
        Normal Graph
    predicate : lambda 
        Predicate over edges to be kept
        in the subgraph
    """
    G_ = copy.deepcopy(G)
    badEdges = list(itertools.filterfalse(lambda e : predicate(G_.edges[e]), G_.edges))
    G_.remove_edges_from(badEdges)
    return G_

def bestAssignmentCost (costTable) :
    """
    Compute the minimum total
    cost assignment.

    Parameters
    ----------
    costTable : dict()
        For each pair, what is the 
        cost of matching that pair
        together.
    """
    G = nx.Graph()

    for key, val in costTable.items() :
        i, j = key
        G.add_node(i, bipartite=0)
        G.add_node(str(j), bipartite=1)
        G.add_edge(i, str(j), weight=val)

    matching = bipartite.minimum_weight_full_matching(G)
    # Each edge gets counted twice
    cost = sum([G.edges[e]['weight'] for e in matching.items()]) / 2
    return cost

def subsetSize(s, t, subSize) :
    """
    Calculate the size of each
    subtree in the rooted tree t.

    Parameters
    ----------
    s : int
        The current vertex
    t : nx.DiGraph()
        The tree
    subSize : dict
        Store the size of the subtree
        rooted at each vertex.
    """
    subSize[s] = 1
    nbrs = list(t.neighbors(s))
    if len(nbrs) != 0 :
        for u in nbrs :
            subsetSize(u, t, subSize)
            subSize[s] += subSize[u]

def match (rt1, rt2) :
    """
    Given two full rooted binary trees,
    find the minimum nodes needed to be 
    added to either of the trees to make
    them isomorphic.

    >>> t1 = nx.DiGraph()
    >>> t2 = nx.DiGraph()
    >>> t1.add_edges_from([(0,1), (1,3), (0,2)])
    >>> t2.add_edges_from([(0,1), (0,2), (2,3)])
    >>> match((0, t1), (0, t2))

    Parameters
    ----------
    rt1 : tuple
        A (root index, nx.DiGraph) tuple representing
        the first rooted tree.
    rt2 : tuple
        The second rooted tree.
    """
    r1, t1 = rt1
    r2, t2 = rt2
    sub1 = dict()
    sub2 = dict()
    subsetSize(r1, t1, sub1)
    subsetSize(r2, t2, sub2)

    def cost (u1, u2) :
        nbrs1 = list(t1.neighbors(u1))
        nbrs2 = list(t2.neighbors(u2))
        degree1 = len(nbrs1)
        degree2 = len(nbrs2)
        if degree1 == 0 and degree2 == 0 :
            return 0
        elif degree1 == 0 : 
            return sub2[u2] - sub1[u1];
        elif degree2 == 0 :
            return sub1[u1] - sub2[u2]
        else :
            prod = list(itertools.product(nbrs1, nbrs2))
            costs = list(map(lambda x : cost(*x), prod))
            nbrs2 = [str(_) for _ in nbrs2]
            costDict = dict(zip(prod, costs))
            return bestAssignmentCost(costDict)

    return cost(r1, r2)

def parallelize(indirList, outdir, function, writer, **kwargs) :
    """ 
    Use multiprocessing library to 
    parallelize preprocessing. 

    A very common operation that I
    have to do is to apply some
    operations on some files in a
    directory and write to some other
    directory. Often, these operations 
    are very slow. This function takes
    advantage of the multiple cpus to 
    distribute the computation.

    Parameters
    ----------
    indirList : list
        paths to the input directories
    outdir : str
        path to the output directory
    function : lamdba 
        function to be applied on 
        each file in the input directory
    writer : lambda
        function which writes the results
        of function application to a file of
        given name
    """

    def subTask (inChunk, outChunk) :
        for i, o in zip(inChunk, outChunk) :
            obj = function(i, **kwargs)
            writer(obj, o, **kwargs)

    cpus = mp.cpu_count()

    names = list(map(lambda x : os.path.splitext(x)[0], os.listdir(indirList[0])))
    names.sort()

    inFullPaths = list(zipDirs(indirList))
    outFullPaths = [os.path.join(outdir, name) for name in names]

    inChunks = more_itertools.divide(cpus, inFullPaths)
    outChunks = more_itertools.divide(cpus, outFullPaths)

    processList = []

    for inChunk, outChunk in zip(inChunks, outChunks) :
        p = mp.Process(target=subTask, args=(inChunk, outChunk,))
        p.start()
        processList.append(p)

    for p in processList :
        p.join()

def rasterize(svgFile, outFile) :
    """
    Utility function to convert svg to
    raster using inkscape

    Parameters
    ----------
    svgFile : str
        Path to input svg
    outFile : str
        Path to where to store output
    """
    subprocess.call(['inkscape', '-h 72', '-w 72', '-z', '-f', svgFile, '-j', '-e', outFile])

def singlePathSvg(path, vb, out) :
    """
    Write a single path to svg file.

    Parameters
    ----------
    path : svg.Document.FlattenedPath
        Path to be written
    vb : list
        The viewbox of the svg
    out : str
        Path to the output file.
    """
    vbox = ' '.join([str(_) for _ in vb])
    wsvg([path[0]], attributes=[path[1].attrib], viewbox=vbox, filename=out)

def getSubsetSvg(paths, lst, vb) :
    """
    An svg is a collection of paths. 
    This function chooses a list of
    paths from the svg and makes an svg
    string for those paths. 

    While doing this, we have to be careful
    about the order in which these paths
    are put because they determine the
    order in which they are rendered. 

    For this, I have tweaked the 
    svgpathtools library to also
    store the zIndex of the paths
    so that we have help while putting
    the paths together

    >>> doc = svg.Document('file.svg')
    >>> paths = doc.flatten_all_paths()
    >>> print(getSubsetSvg(paths, [1,2,3], doc.get_viewbox()))

    Parameters
    ----------
    paths : list
        List where each element is of
        the type svg.Document.FlattenedPath
    lst : list
        An index set into the previous list 
        specifying the paths we want.
    vb : list
        The viewbox of the original svg.
    """
    vbox = ' '.join([str(_) for _ in vb])
    ps = [paths[i][0] for i in lst]
    attrs = [paths[i][1].attrib for i in lst]
    order = [paths[i][3] for i in lst]

    cmb = list(zip(order, ps, attrs))
    cmb.sort()

    ps = [p for _, p, _ in cmb]
    attrs = [a for _, _, a in cmb]

    drawing = svg.disvg(ps, 
                        attributes=attrs, 
                        viewbox=vbox, 
                        paths2Drawing=True, 
                        openinbrowser=False)
    return drawing.tostring()

class GraphReadWrite () :
    """
    Convenience class to handle graph
    read/write operations so that
    I don't have to keep searching 
    for networkx documentation.

    Two graph types are supported right now:
        1) 'cytoscape'
        2) 'tree'
    """

    def __init__ (self, graphType) :
        """
        Parameters
        ----------
        graphType : str
            Indicate which graph type
            to read/write.
        """
        self.graphType = graphType

    def read (self, inFile) :
        """
        Parameters
        ----------
        inFile : str
            Path to input file to write to.
        """
        with open(inFile, 'r') as fd :
            dct = json.loads(fd.read())
        if self.graphType == 'cytoscape' :
            return nx.readwrite.json_graph.cytoscape_graph(dct)
        elif self.graphType == 'tree' :
            return nx.readwrite.json_graph.tree_graph(dct)
        
        raise ValueError ('Unsupported Graph Type')

    def write (self, G, outFile) :
        """
        Parameters
        ----------
        G : nx.Graph / (nx.DiGraph, int)
            If graphType is cytoscape then
            it is a nx.Graph, else, it is the 
            tree along with the root index.
        outFile : str
            Path to the output file.
        """
        dct = None
        if self.graphType == 'cytoscape' :
            dct = nx.readwrite.json_graph.cytoscape_data(G)
        elif self.graphType == 'tree' :
            T, r = G
            dct = nx.readwrite.json_graph.tree_data(T, r)

        if dct is None :
            raise ValueError ('Unsupported Graph Type')

        with open(outFile, 'w+', encoding='utf-8') as fd :
            json.dump(dct, fd, ensure_ascii=False, indent=4)

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

    H = pts1_.T @ pts2_

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
            R[:, 1] *= -1

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
        
def complexLE (x, y) :
    """
    Lesser than equal to for complex numbers.

    Parameters
    ----------
    x : complex
    y : complex
    """
    if x.real < y.real :
        return True
    elif x.real > y.real :
        return False
    else :
        return x.imag < y.imag

def complexDot (x, y) :
    """
    Dot product between complex
    numbers treating them as vectors
    in R^2.

    Parameters
    ----------
    x : complex
    y : complex
    """
    return (x.real * y.real) + (x.imag * y.imag)

def complexCross (x, y) : 
    """
    Cross product between complex
    numbers treating them as vectors
    in R^2.

    Parameters
    ----------
    x : complex
    y : complex
    """
    v1 = np.array([x.real, x.imag, 0.0]) 
    v2 = np.array([y.real, y.imag, 0.0]) 
    return np.linalg.norm(np.cross(v1,v2))

    
def d2 (path, docbb, bins=10, nSamples=100) :
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

    return hist 

def shell(path, docbb, bins=10) :
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

def a3 (path, docbb, bins=10, nSamples=100) :
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

def d3 (path, docbb, bins=10, nSamples=100) :
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

def fd (path, docbb, nSamples=100) :
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
    centroid = np.mean(pts)
    rt = np.abs(pts - centroid)    
    an = np.fft.fft(rt) / nSamples
    bn = np.abs(an / an[0])
    return bn[:min(nSamples,20)].tolist()
    
def relbb (path, docbb) :
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

    return [x1, x2, y1, y2]

def symGraph (paths) : 
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

def adjGraph (paths) :
    """
    Return a complete graph over the 
    paths. We have course estimates
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

def graphFromSvg (svgFile, graphFunction) : 
    """
    Produce a relationship graph for 
    the svg file

    Parameters
    ----------
    svgFile : str
        path to svg file.
    graphFunction : lambda 
        path relation computer.
    """
    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    G = graphFunction(paths)
    return G

def relationshipGraph (svgFile, relFunctions) :
    """
    Take the svg, find graphs
    for all relationship types
    and combine them into one single
    graph for clustering.

    Parameters
    ----------
    svgFile : str
        path to svg file.
    relFunctions : list
        List of graph computing functions
    """
    graphs = map(lambda x : graphFromSvg(svgFile, x), relFunctions)
    graph = None
    for g in graphs :
        if graph is None:
            graph = g
        else :
            graph = nx.compose(graph, g)
    return graph

def treeImageFromGraph (G) :
    """
    Visualize the paths as depicted
    by the tree structure of the 
    graph.

    This helps us understand whether the 
    network's decomposition is good or
    not.

    Parameters
    ----------
    G : nx.DiGraph
        Hierarchy of paths
    """
    fig, ax = plt.subplots(dpi=1500)
    pos = graphviz_layout(G, prog='dot')
    ax.set_aspect('equal')
    nx.draw(G, pos, ax=ax, node_size=0.5, arrowsize=1)
    for n in G :
        img = svgStringToBitmap(G.nodes[n]['svg'])
#        imagebox = OffsetImage(img, zoom=0.2)
        imagebox = OffsetImage(img, zoom=0.08)
        imagebox.image.axes = ax
        ab = AnnotationBbox(imagebox, pos[n], pad=0)
        ax.add_artist(ab)
    ax.axis('off')
    return (fig, ax)

def treeImageFromJson (jsonTuple) :
    """
    Given a tuple containing
    a single json file representing the
    tree structure of a particular svg,
    we visualize the figure using 
    matplotlib.

    For each rooted subtree, we keep an
    icon depicting the paths in that 
    subtree.

    This helps us understand whether the 
    network's decomposition is good or
    not.

    Parameters
    ----------
    jsonTuple : tuple
        singleton tuple containing path 
        to the json file containing
        tree data.
    """
    jsonFile, = jsonTuple
    G = GraphReadWrite('tree').read(jsonFile)
    return treeImageFromGraph(G)

def hik (a, b) :
    """
    Histogram Intersection Kernel.

    A distance measure between two
    histograms.

    Parameters
    ----------
    a : np.ndarray
        The first histogram.
    b : np.ndarray
        The second histogram.
    """
    return np.sum(np.minimum(a, b))

def chi2 (a, b) :
    """
    Chi-squared distance between
    two vectors.

    Parameters
    ----------
    a : np.ndarray
        The first vector.
    b : np.ndarray
        The second vector.
    """
    n = (a - b) * (a - b) 
    d = a + b
    n[d > 0] /= d[d > 0]
    return np.sum(n)

def bhattacharya (a, b) :
    """
    Bhattacharya distance between
    two vectors.

    Parameters
    ----------
    a : np.ndarray
        The first vector.
    b : np.ndarray
        The second vector.
    """
    return 1 - np.sum(np.sqrt(a * b))

def putOnCanvas (pts, images, outFile) :
    """ 
    Put images at the corresponding
    points in R^2 in a super-large
    canvas.

    Mainly for tSNE visualization.
    
    The images should have an alpha
    channel.

    Parameters
    ----------
    pts : np.ndarray
        Collection of 2D points
    images : list
        Corresponding images to embed
    outFile : str
        Path to output file.
    """
    min_x = np.min(pts[:,0])
    max_x = np.max(pts[:,0])
   
    min_y = np.min(pts[:,1])
    max_y = np.max(pts[:,1])

    h, w, _ = images[0].shape

    sz = 10000
    pad = (h + w)
    pix = sz + 2 * pad
    canvas = np.ones((pix, pix, 4), dtype=np.uint8) * int(255)
    canvas[:, :, 3] = 0
    
    for pt, im in zip(pts, images) : 
        h_, w_, _ = im.shape
        if h_ != h or w_ != w :
            im = transform.resize(im, (h, w))
        pix_x = pad + math.floor(sz * ((pt[0] - min_x) / (max_x - min_x)))
        pix_y = pad + math.floor(sz * ((pt[1] - min_y) / (max_y - min_y)))
        
        sx, ex = pix_x - (h // 2), pix_x + (h // 2)
        sy, ey = pix_y - (w // 2), pix_y + (w // 2)
        
        canvas[sx:ex, sy:ey,:] = im
    
    imwrite(outFile, canvas)

def graphCluster (G, algo, doc, descFunctions) :
    """
    Hierarchical clustering of a graph into
    a dendogram. Given an algorithm to 
    partition a graph into two sets, the
    this class produces a tree by recursively 
    partitioning the graphs induced on these
    subsets till each subset contains only a
    single node.

    Parameters
    ----------
    G : nx.Graph
        The path relationship graph
    algo : lambda
        The partitioning algorithm
        to be used
    doc : svg.Document
        Used to set node attributes.
    descFunctions : list
        List of descriptor functions
        to compute for each leaf node.
    """

    def cluster (lst) :
        """
        Recursively cluster the
        subgraph induced by the 
        vertices present in the 
        list.

        lst : list
            List of vertices to
            be considered.
        """
        tree.add_node(idxDct['idx'])
        curId = idxDct['idx']
        idxDct['idx'] += 1
        if len(lst) != 1 :
            subgraph = G.subgraph(lst)
            l, r = algo(subgraph)
            lId = cluster(list(l))
            rId = cluster(list(r))
            tree.add_edge(curId, lId)
            tree.add_edge(curId, rId)
            # Add indices of paths in this subtree.
            tree.nodes[curId]['pathSet'] = lst
        else :
            pathId = lst.pop()
            tree.nodes[curId]['pathSet'] = [pathId]
            # Add descriptors
            tree.nodes[curId]['desc'] = more_itertools.collapse(
                [f(paths[pathId].path, vbox) for f in descFunctions]
            )

        # Add svg
        tree.nodes[curId]['svg'] = getSubsetSvg(
            paths, 
            tree.nodes[curId]['pathSet'], 
            vbox
        )
        # Add image
        # tree.nodes[curId]['img'] = svgStringToBitmap(tree.nodes[curId]['svg'])
        return curId

    paths = doc.flatten_all_paths()
    vbox = doc.get_viewbox()
    tree = nx.DiGraph()
    idxDct = {'idx' : 0}
    root = 0
    cluster(list(G.nodes))
    return tree, root

def treeApply (T, r, function) : 
    """
    Apply function to all nodes in the
    tree.

    Parameters
    ----------
    T : nx.DiGraph
        The tree.
    r : object
        Root of the tree.
    function : lambda
        A function which takes as
        input the tree, current node 
        and performs some operation.
    """
    for child in T.neighbors(r) :
        treeApply(T, r, function)

    function(T, r, T.neighbors(r))

def randomString(k) : 
    """
    Return a random string of lower case
    alphabets.

    Parameters
    ----------
    k : int
        Length of the random string.
    """
    alphabets = string.ascii_lowercase
    return ''.join(random.choices(alphabets, k=k))

def svgStringToBitmap (svgString) :
    svgName = randomString(10) + '.svg'
    svgName = osp.join('/tmp', svgName)

    pngName = randomString(10) + '.png'
    pngName = osp.join('/tmp', pngName)

    with open(svgName, 'w+') as fd :
        fd.write(svgString)

    rasterize(svgName, pngName)
    img = image.imread(pngName)

    os.remove(svgName)
    os.remove(pngName)

    return img.tolist()

def matplotlibFigureSaver (obj, fname) :
    """ 
    Save matplotlib figures.

    Parameters
    ----------
    obj : tuple
        A tuple of plt.Figure, plt.Axes
    fname : str
        Path to output file.
    """
    fig, _ = obj
    fig.savefig(fname + '.png')
    plt.close(fig)

def findRoot (tree) :
    """
    Find the root of a tree.

    Parameters
    ----------
    tree : nx.DiGraph
        Rooted tree with unknown root.
    """
    return next(nx.topological_sort(tree))

def configReadme (path) :
    """
    I think it is a good idea to save the 
    current snapshot of the config 
    file along with the trained model.

    Parameters
    ----------
    path : str
        Path to save to.
    """
    with open('Config.py', 'r') as fd: 
        content = fd.read()
    with open(path, 'w+') as fd :
        fd.write('Configuration used:\n')
        fd.write('```\n')
        fd.write(content)
        fd.write('```\n')

def getTreeStructureFromSVG (svgFile) : 
    """
    Infer the tree structure from the
    XML document. 

    Parameters
    ----------
    svgFile : str
        Path to the svgFile.
    """

    def skeletonTree(tree) :
        """
        Remove all elements which were 
        going to be rendered. These are 
        paths, lines, circles etc. 

        Parameters 
        ----------
        tree : ElementTree
            The parsed XML tree.
        """
        skeleton = copy.deepcopy(tree)
        root = skeleton.getroot()
        for tag in relevantTags : 
            allTags = root.findall(tag)
            for instance in allTags :
                root.remove(instance)
        return skeleton

    def buildTreeGraph (element, svgString) :
        """
        Recursively create networkx tree and 
        add svg strings to the tree 
        corresponding to the SVG document.

        Parameters
        ----------
        element : Element
            Element at this level of the
            tree.
        svgString : str
            String representation of this
            element.
        """
        curId = r['idx']
        T.add_node(curId, svg=svgString)
        for child in element : 
            if child.tag in relevantTags : 
                r['idx'] += 1
                childId = r['idx']
                skeletonCopy = copy.deepcopy(skeleton)
                skeletonCopy.getroot().append(child)
                elementStr = str(ET.tostring(skeletonCopy.getroot()), 'utf-8')
                buildTreeGraph(child, elementStr)
                T.add_edge(curId, childId)

    relevantTags = [
        '{http://www.w3.org/2000/svg}rect',
        '{http://www.w3.org/2000/svg}circle',
        '{http://www.w3.org/2000/svg}ellipse',
        '{http://www.w3.org/2000/svg}line', 
        '{http://www.w3.org/2000/svg}polyline',
        '{http://www.w3.org/2000/svg}polygon',
        '{http://www.w3.org/2000/svg}path',
        '{http://www.w3.org/2000/svg}g'
    ]

    tree = ET.parse(svgFile)
    root = tree.getroot()
    skeleton = skeletonTree(tree)

    T = nx.DiGraph()
    r = {'idx' : 0} 

    rootString = str(ET.tostring(root), 'utf-8')
    buildTreeGraph (root, rootString)
    return T

def listdir (path) :
    """
    Convenience function to get 
    full path details while calling os.listdir

    Parameters
    ----------
    path : str
        Path to be listed.
    """
    return [osp.join(path, f) for f in os.listdir(path)]

def comp (fileTuple) : 
    fileName, = fileTuple
    return treeImageFromGraph(svgToTree(fileName))

if __name__ == "__main__" : 
    parallelize(['./Avatar/Train/'], './Avatar/Hierarchies/Train/', comp, matplotlibFigureSaver)
