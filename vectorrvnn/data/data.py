from functools import reduce
import svgpathtools as svg
from skimage import transform
import networkx as nx
import numpy as np
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

    def initGraphic (self, doc) :
        self.doc = doc
        paths = cachedPaths(self.doc)
        paths = [p for i, p in enumerate(paths) 
                if not pathBBox(p.path).isDegenerate()]
        self.nPaths = len(paths)
        self._computeBBoxes(findRoot(self))

    def _setPathSets (self) : 
        """ set pathsets for each node """
        for n in self.nodes :
            self.nodes[n]['pathSet'] = tuple(leavesInSubtree(self, n))

    def __init__ (self, svgFile=None, treePickle=None, tree=None) : 
        """ only one of treePickle and tree can be not None """
        assert(treePickle is None or tree is None)
        if treePickle is not None : 
            super(SVGData, self).__init__(nx.read_gpickle(treePickle))
        elif tree is not None : 
            super(SVGData, self).__init__(tree)
        else :
            super(SVGData, self).__init__()
        self.initTree()
        self.svgFile = svgFile
        self.initGraphic(svg.Document(svgFile))

    def _computeBBoxes (self, node) : 
        paths = [p.path for p in cachedPaths(self.doc)]
        for n in self.nodes : 
            ps = self.nodes[n]['pathSet']
            relPaths = [paths[i] for i in ps]
            bbox = reduce(lambda x, y: x + y, map(pathBBox, relPaths))
            nx.set_node_attributes(self, {n: bbox}, 'bbox')
    
