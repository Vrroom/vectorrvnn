from functools import reduce
import svgpathtools as svg
from skimage import transform
import networkx as nx
import numpy as np
from subprocess import call
from vectorrvnn.geometry import *
from vectorrvnn.utils import *

class SVGData (nx.DiGraph) : 
    """
    Each SVG data point is a tree along with an svg.Document 
    containing the geometric information.

    The tree is preprocessed to include the subtree sizes,
    the node depth (measured from the top and the bottom)
    and the path sets at each node. 
    """
    def initTree (self, tree=None) : 
        if not nx.is_tree(self) : 
            tree = forest2tree(self)
        if tree is not None : 
            self.clear()
            self.update(tree)
        setSubtreeSizes(self)
        setNodeDepths(self)
        setNodeBottomDepths(self)
        self._setPathSets()
        assert nx.is_tree(self)

    def initGraphic (self) :
        paths = cachedPaths(self.doc)
        self.nPaths = len(paths)
        assert(self.nPaths == len(leaves(self)))
        box = getDocBBox(self.doc) 
        tol = max(box.w, box.h) / 100
        self.lines = [flattenPath(p.path, tol) for p in paths]
        self._computeBBoxes()
        self._computeOBBs()
        self._computeColors()

    def _computeColors(self) : 
        self.fills, self.strokes = [], []
        for p in cachedPaths(self.doc) : 
            self.fills.append(pathColorFeature(p, 'fill'))
            self.strokes.append(pathColorFeature(p, 'stroke'))

    def _computeOBBs (self) : 
        self.obbs = []
        for lines in self.lines :
            self.obbs.append(polylineOBB(self.doc, lines))

    def _computeBBoxes (self) : 
        self.bbox = []
        for lines in self.lines :
            self.bbox.append(polylineAABB(self.doc, lines))

    def recalculateBBoxes(self, fn) : 
        for i in range(len(self.bbox)) : 
            self.bbox[i] = fn(self.bbox[i])

    def _setPathSets (self) : 
        """ set pathsets for each node """
        for n in self.nodes :
            self.nodes[n]['pathSet'] = tuple(leavesInSubtree(self, n))

    def __init__ (self, svgFile=None, treePickle=None, tree=None, convert2usvg=False) : 
        """ only one of treePickle and tree can be not None """
        assert(treePickle is None or tree is None)
        if convert2usvg :
            newName = f'/tmp/{getBaseName(svgFile)}.svg'
            call(['usvg', svgFile, newName])
            svgFile = newName
        with open(svgFile) as fp:
            self.svg = fp.read()
        self.doc = withoutDegeneratePaths(svg.Document(svgFile))
        if treePickle is not None : 
            super(SVGData, self).__init__(nx.read_gpickle(treePickle))
        elif tree is not None : 
            super(SVGData, self).__init__(tree)
        else :
            super(SVGData, self).__init__(getTreeStructureFromSVG(self.doc))
        self.initTree()
        self.svgFile = svgFile
        self.initGraphic()

    def __or__ (self, that) : 
        """
        Take the union of two datapoints. Compositing one on
        top of the other.
        """
        dataUnion = deepcopy(self) 
        t1, t2 = nx.DiGraph(self), nx.DiGraph(that)
        # combine the trees of self and that
        dataUnion.initTree(treeUnion(t1, t2))
        # combine the documents 
        dataUnion.doc = docUnion(self.doc, that.doc) 
        # new paths are the sum of individual paths
        dataUnion.nPaths = self.nPaths + that.nPaths
        # set the bbox for the new root
        newRoot = findRoot(dataUnion)
        leafBoxes = self.bbox + that.bbox
        return dataUnion

    def __lt__ (self, that) : 
        return repr(self.doc) < repr(that.doc)
