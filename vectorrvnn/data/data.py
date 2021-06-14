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
        if tree is not None : 
            self.clear()
            self.update(tree)
        setSubtreeSizes(self)
        setNodeDepths(self)
        setNodeBottomDepths(self)
        self._setPathSets()

    def initGraphic (self, doc) :
        self.doc = doc
        paths = cachedPaths(self.doc)
        paths = [p for i, p in enumerate(paths) 
                if not isDegenerateBBox(bb(self.doc, i))]
        self.nPaths = len(paths)
        self.pathViewBoxes = [bb(self.doc, i) for i, p in enumerate(paths)]
        for r in [r for r in self.nodes if self.in_degree(r) == 0] : 
            self._computeBBoxes(r)

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
        if self.out_degree(node) == 0 : 
            pId = self.nodes[node]['pathSet'][0]
            nx.set_node_attributes(self, {node: self.pathViewBoxes[pId]}, 'bbox')
        else : 
            for n in self.neighbors(node) : 
                self._computeBBoxes(n)
            boxes = np.array([self.nodes[n]['bbox'] for n in self.neighbors(node)])
            xm, ym = boxes[:,0].min(), boxes[:,1].min()
            xM, yM = (boxes[:,0] + boxes[:,2]).max(), (boxes[:,1] + boxes[:,3]).max()
            nx.set_node_attributes(self, {node: [xm, ym, xM - xm, yM - ym]}, 'bbox')
    
