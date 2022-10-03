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
        paths = self.doc.paths()
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

    def recalculateBoxes(self) : 
        self._computeBBoxes()
        self._computeOBBs()

    def _setPathSets (self) : 
        """ set pathsets for each node """
        for n in self.nodes :
            self.nodes[n]['pathSet'] = tuple(leavesInSubtree(self, n))

    def __init__ (self, svgFile, treePickle=None, tree=None, convert2usvg=False, normalize=False) : 
        """
        Data structure containing geometry and topology of the graphic

        This data structure can be instantiated in a few ways. An svg file
        is compulsory. After that, the topology can be supplied through a 
        pickle file or a networkx graph (using arguments treePickle and tree). 
        At most one of these arguments should be supplied. If both are not specified
        then, we infer the topology from the markup (e.g. <g/>) in the svg document.

        convert2usvg simplifies the svg before extracting geometry so that we have
        fewer cases to worry about.

        normalize sets the graphic in a fixed viewbox for consistency. 
        """ 
        # First initialize the topology of the graphic
        assert(treePickle is None or tree is None)
        if treePickle is not None : 
            super(SVGData, self).__init__(nx.read_gpickle(treePickle))
        elif tree is not None : 
            super(SVGData, self).__init__(tree)
        else :
            doc = withoutDegeneratePaths(svg.Document(svgFile))
            super(SVGData, self).__init__(getTreeStructureFromSVG(doc))
        self.initTree()

        # Read the original svg file. Easier for debugging
        with open(svgFile) as fp:
            self.svg = fp.read()
        self.svgFile = svgFile
        
        # option to simplify svg - helps with wild svgs
        if convert2usvg :
            newName = f'/tmp/{getBaseName(svgFile)}.svg'
            call(['usvg', svgFile, newName])
            svgFile = newName

        # option to normalize svg - like I do in training
        if normalize : 
            newName = f'/tmp/{getBaseName(svgFile)}.svg'
            doc = svg.Document(svgFile)
            box = union([pathBBox(p.path) for p in doc.paths()])
            setDocBBox(doc, box.normalized() * 1.2)
            with open(newName, 'w+') as fp : 
                fp.write(repr(doc))
            svgFile = newName
    
        self.doc = withoutDegeneratePaths(svg.Document(svgFile))
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
