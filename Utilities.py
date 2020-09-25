# Avoid repetition at all costs.
import alphashape
import torch_geometric.data as tgd
import torch
import pickle
import os
import os.path as osp
import time
import json
import string
import subprocess
import more_itertools
import itertools
from functools import partial
from functools import reduce
from tqdm import tqdm
import networkx as nx
import xml.etree.ElementTree as ET
from networkx.drawing.nx_agraph import graphviz_layout
from networkx.algorithms.community import kernighan_lin_bisection
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
import Data
from scipy.spatial import ConvexHull, distance_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def area (bbox) : 
    """
    Compute the area of the bounding box.
    """
    bbox = bbox.view((-1, 4))
    return (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

def iou (bbox1, bbox2) : 
    """
    Compute the intersection over union for two bounding
    boxes.

    Assume that there are N querying boxes and N target
    boxes. Hence the inputs have shape N x 4. The first
    two columns contain the coordinate of the top-left 
    corner and the last two contain the coordinates of
    the bottom-right corner.

    Parameters
    ----------
    bbox1 : torch.tensor
        Query bounding box.
    bbox2 : torch.tensor
        Target bounding box.
    """
    bbox1, bbox2 = bbox1.view((-1, 4)), bbox2.view((-1, 4))
    xMin, yMin = max(bbox1[:, 0], bbox2[:, 0]), max(bbox1[:, 1], bbox2[:, 1])
    xMax, yMax = min(bbox1[:, 2], bbox2[:, 2]), min(bbox1[:, 3], bbox2[:, 3])
    intersection = (xMax - xMin) * (yMax - yMin)
    union = area(bbox1) + area(bbox2)
    return intersection / union
    
def dictionaryProduct (d) : 
    product = itertools.product
    keys = d.keys()
    vals = map(lambda v : v if isinstance(v, list) else [v], d.values())
    combinations = product([keys], product(*vals))
    dicts = map(lambda items : dict(zip(*items)), combinations)
    return dicts

def plotDistanceMatrix (f) : 
    dists = distance_matrix(f, f)
    plt.imshow(dists)
    plt.show()

def plotTSNE (x, y, k, perp) : 
    pca = TSNE(n_components=2, perplexity=perp)
    out = pca.fit_transform(x)
    colors = [(1,0,0,0.5), (0, 0, 1, 0.5)]
    for i in range(k) : 
        pts = out[y == i]
        plt.scatter(pts[:, 0], pts[:, 1], color=colors[i])
    plt.show()

def plotPCA (x, y, k) : 
    pca = PCA(n_components=2)
    out = pca.fit_transform(x)
    colors = ['red', 'blue']
    for i in range(k) : 
        pts = out[y == i]
        # color = (np.random.rand(), np.random.rand(), np.random.rand(), 0.2)
        plt.scatter(pts[:, 0], pts[:, 1], color=colors[i])
    plt.show()

def aggregateDict (listOfDicts, reducer) : 
    keys = set(more_itertools.flatten(map(lambda x : x.keys(), listOfDicts)))
    aggregator = lambda key : reducer(map(lambda x : x[key], listOfDicts))
    return dict(zip(keys, map(aggregator, keys)))

def tree2Document (document, tree, attribs) :
    """
    Add a tree of parents to svg document.

    Parameters
    ----------
    document : svg.Document
        SVG document to which the tree of paths 
        has to be added.
    tree : nx.DiGraph
        Path hierarchy.
    attribs : dict
        Dictionary of path attributes. Same attributes
        are applied to all paths.
    """
    def addToDocument (T, r, children) :
        if T.out_degree(r) == 0 : 
            document.add_path(T.nodes[r]['path'], attribs=attribs, group=parentGroup[r])
        else :
            newGroup = document.add_group(parent=parentGroup[r])
            for child in children: 
                parentGroup[child] = newGroup

    root = findRoot(tree)
    parentGroup = {root: None}
    treeApplyRootFirst(tree, root, addToDocument)
    return document

class ComposeAdd () : 

    def __init__ (self, descFunctions) :
        self.descFunctions = descFunctions
    
    def __call__ (self,  p, b, **kwargs) : 
        return reduce(lambda x, y : x + y(p, b, **kwargs), self.descFunctions, []) 

def isBBoxDegenerate(bbox) :
    """
    Check if the bounding box has
    zero area.

    Parameters
    ----------
    bbox : tuple
        The top left and bottom right
        corners of the box.
    """
    xmin, xmax, ymin, ymax = bbox
    return xmin == xmax and ymin == ymax

def cluster2DiGraph (jsonFile, svgFile) :
    """
    Convert the output of the svg-app 
    clustering app into a digraph.
    """
    with open(jsonFile, 'r') as fd :
        data = json.load(fd)

    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    paths = list(filter(lambda p : not isBBoxDegenerate(p.path.bbox()), paths))

    tree = nx.DiGraph()
    tree.add_nodes_from(range(len(data["nodes"])))

    for node in data["nodes"] : 
        tree.nodes[node["id"]]["pathSet"] = node["paths"]
        for child in node["children"] :
            tree.add_edge(node["id"], child)

    return Data.Tree(tree)

def levenshteinDistance (a, b, matchFn, costFn) : 
    """
    Calculate the optimal tree edit distance
    between two lists, given the costs of
    matching two items on the list and 
    the costs of deleting an item in the
    optimal match.

    Example
    -------
    >>> skipSpace = lambda x : 0 if x == ' ' else 1
    >>> match = lambda a, b : 0 if a == b else 1
    >>> print(levenshteinDistance('sumitchaturvedi', 'sumi t ch at urvedi', match, skipSpace))

    In the above example, since we don't penalize for 
    deleting spaces, the levenshtein distance is 
    going to be 0.

    Parameters
    ----------
    a : list
        First string.
    b : list
        Second string.
    matchFn : lambda
        The cost of matching the
        ith character in a with the 
        jth character in b.
    costFn : lambda 
        The penalty of skipping over
        one character from one of the
        strings.
    """
    n = len(a)
    m = len(b)
    row = list(range(n + 1))

    for j in range(1, m + 1) : 
        newRow = [0 for _ in range(n + 1)]
        for i in range(n + 1): 
            if i == 0 : 
                newRow[i] = j
            else : 
                match = matchFn(a[i-1], b[j-1])
                cost1 = costFn(a[i-1])
                cost2 = costFn(b[j-1])
                newRow[i] = min(row[i-1] + match, newRow[i-1] + cost1, row[i] + cost2)
        row = newRow
    return row[n]

def svgTreeEditDistance (t1, t2, paths, vbox) :
    """
    Compute two Tree objects (which capture)
    SVG document structure. Each node in the 
    tree is a group of paths. We have a 
    similarity function between these nodes 
    based on the levenshtein distance between the 
    path strings.
    
    Parameters
    ----------
    t1 : Data.Tree
        Tree one.
    t2 : Data.Tree
        Tree two.
    paths : list
        List of paths.
    vbox : list
        Bounding Box.
    """

    def pathMatchFn (a, b) : 

        def curveMatchFn (a, b) :
            """
            If the curve parameters match
            within a distance threshold, then
            we let them be.
            """
            if isinstance(a, type(b)) : 
                if isinstance(a, svg.Line) :
                    cond1 = abs(a.start - b.start) < thresh
                    cond2 = abs(a.end   - b.end  ) < thresh
                    return 0 if cond1 and cond2 else 1
                elif isinstance(a, svg.QuadraticBezier) : 
                    cond1 = abs(a.start   - b.start  ) < thresh
                    cond2 = abs(a.end     - b.end    ) < thresh
                    cond3 = abs(a.control - b.control) < thresh
                    return 0 if cond1 and cond2 and cond3 else 1
                elif isinstance(a, svg.CubicBezier) : 
                    cond1 = abs(a.start    - b.start   ) < thresh
                    cond2 = abs(a.end      - b.end     ) < thresh
                    cond3 = abs(a.control1 - b.control1) < thresh
                    cond4 = abs(a.control2 - b.control2) < thresh
                    return 0 if cond1 and cond2 and cond3 and cond4 else 1
                elif isinstance(a, svg.Arc) : 
                    cond1 = abs(a.start  - b.start ) < thresh
                    cond2 = abs(a.end    - b.end   ) < thresh
                    cond3 = abs(a.radius - b.radius) < thresh
                    return 0 if cond1 and cond2 and cond3 else 1
                else :
                    raise TypeError
            else :
                return 1

        def curveDelFn (a) : 
            return 1 
    
        if (a, b) not in cachedMatchVals : 
            pathA = paths[a].path
            pathB = paths[b].path
            maxLen = max(len(pathA), len(pathB))

            pathMatch = levenshteinDistance(pathA, pathB, curveMatchFn, curveDelFn)
            normalized = pathMatch / maxLen

            cachedMatchVals[(a, b)] = normalized
            cachedMatchVals[(b, a)] = normalized
        
        return cachedMatchVals[(a, b)]

    def pathDelFn (a) : 
        return 1

    def cost (u1, u2) :
        
        def pathSetDist (x, y) : 
            ps1 = t1.tree.nodes[x]['pathSet']
            ps2 = t2.tree.nodes[y]['pathSet']
            lev = levenshteinDistance(ps1, ps2, pathMatchFn, pathDelFn)
            maxLen = max(len(ps1), len(ps2))
            return lev / maxLen

        nbrs1 = list(t1.tree.neighbors(u1))
        nbrs2 = list(t2.tree.neighbors(u2))
        degree1 = t1.tree.out_degree(u1)
        degree2 = t2.tree.out_degree(u2)
        if degree1 == 0 and degree2 == 0 :
            return pathSetDist(u1, u2)
        elif degree1 == 0 : 
            return sub2[u2] - sub1[u1];
        elif degree2 == 0 :
            return sub1[u1] - sub2[u2]
        else :
            prod = list(itertools.product(nbrs1, nbrs2))
            costs = list(map(lambda x : cost(*x) + pathSetDist(*x), prod))
            nbrs2 = [str(_) for _ in nbrs2]
            costdict = dict(zip(prod, costs))
            return bestAssignmentCost(costdict)

    
    xmin, ymin, xmax, ymax = vbox
    diagonal = math.sqrt((xmax - xmin)**2 + (ymax - ymin)**2)
    thresh = diagonal / 10

    cachedMatchVals = dict() 
    sub1, sub2 = dict(), dict()
    subtreeSize(t1.root, t1.tree, sub1)
    subtreeSize(t2.root, t2.tree, sub2)
    c = cost(t1.root, t2.root)
    return c

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
    subsetsize(r1, t1, sub1)
    subsetsize(r2, t2, sub2)

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
            costdict = dict(zip(prod, costs))
            return bestAssignmentCost(costdict)

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

def SVGtoNumpyImage (svgFilePath, H, W) :
    """
    Take an SVG file and rasterize it to 
    obtain a numpy array of given height and
    width. 

    Parameters
    ----------
    svgFilePath : str
    H : float
        Desired height of output.
    W : float 
        Desired width of output.
    """
    rs = randomString(10)
    rasterize(svgFilePath, f'/tmp/{rs}.png', H=H, W=W)
    img = image.imread(f'/tmp/{rs}.png')
    return img[:, :, :3]

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
    G = graphFunction(paths, vbox=doc.get_viewbox())
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
        imagebox = OffsetImage(img, zoom=0.2)
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

def graphCluster (G, algo, doc) :
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
        nonlocal idx
        tree.add_node(idx)
        curId = idx
        idx += 1
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

        return curId

    paths = doc.flatten_all_paths()
    vbox = doc.get_viewbox()
    tree = nx.DiGraph()
    idx = 0
    root = 0
    cluster(list(G.nodes))
    return tree, root

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

def getTreeStructureFromSVG (svgFile) : 
    """
    Infer the tree structure from the
    XML document. 

    Parameters
    ----------
    svgFile : str
        Path to the svgFile.
    """

    def buildTreeGraph (element) :
        """
        Recursively create networkx tree and 
        add path indices to the tree.

        Parameters
        ----------
        element : Element
            Element at this level of the
            tree.
        """
        nonlocal r
        curId = r
        r += 1
        T.add_node(curId)
        if element.tag in childTags : 
            zIdx = allNodes.index(element)
            pathIdx = zIndexMap[zIdx]
            T.nodes[curId]['pathSet'] = [pathIdx]
        else : 
            childIdx = []
            validTags = lambda x : x.tag in childTags or x.tag == groupTag
            children = list(map(buildTreeGraph, filter(validTags, element)))
            T.add_edges_from(list(itertools.product([curId], children)))
            pathSet = list(more_itertools.collapse([T.nodes[c]['pathSet'] for c in children]))
            T.nodes[curId]['pathSet'] = pathSet
        return curId

    childTags = [
        '{http://www.w3.org/2000/svg}rect',
        '{http://www.w3.org/2000/svg}circle',
        '{http://www.w3.org/2000/svg}ellipse',
        '{http://www.w3.org/2000/svg}line', 
        '{http://www.w3.org/2000/svg}polyline',
        '{http://www.w3.org/2000/svg}polygon',
        '{http://www.w3.org/2000/svg}path',
    ]
    groupTag = '{http://www.w3.org/2000/svg}g'

    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    zIndexMap = dict([(p.zIndex, i) for i, p in enumerate(paths)])
    tree = doc.tree
    root = tree.getroot()
    allNodes = list(root.iter())
    T = nx.DiGraph()
    r = 0
    buildTreeGraph (root)
    T = removeOneOutDegreeNodesFromTree(T)
    return Data.Tree(T)

def listdir (path) :
    """
    Convenience function to get 
    full path details while calling os.listdir

    Parameters
    ----------
    path : str
        Path to be listed.
    """
    paths = [osp.join(path, f) for f in os.listdir(path)]
    paths.sort()
    return paths

class AllPathDescriptorFunction () : 
    """ 
    Given a descriptor function, convert it into
    one that acts on all paths
    """

    def __init__ (self, descFunction, **kwargs) : 
        self.descFunction = descFunction
        self.model = None
        if 'model' in kwargs : 
            self.model = kwargs['model']

    def __call__ (self, paths, vb) : 
        descriptors = [self.descFunction(p.path, vb) for p in paths]
        if self.model : 
            descriptors = torch.tensor(descriptors)
            edges = Pose.boneConnectivity
            pathTypes = [p.element.attrib['id'] for p in paths]
            pathTypeIdx = dict(map(reversed, enumerate(pathTypes)))
            targets = torch.tensor([pathTypeIdx[t] for t in pathTypes]).long()
            edges = [(pathTypeIdx[a], pathTypeIdx[b]) for a, b in edges]
            edges = torch.t(torch.tensor(edges).long())
            data = tgd.Data(x=descriptors, edge_index=edges, y=targets)
            return self.model.encoder(data.x, data.edge_index).detach().numpy()
        else :
            return np.array(descriptors)

if __name__ == "__main__" : 
    doc = svg.Document('PartNetSubset/Train/10007.svg')
    paths = doc.flatten_all_paths()
    vbox = doc.get_viewbox()
    G = subgraph(areaGraph(paths, vbox=vbox), lambda x : x['weight'] > 0)
