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
        self.doc = withoutDegeneratePaths(doc)
        paths = cachedPaths(self.doc)
        self.nPaths = len(paths)
        assert(self.nPaths == len(leaves(self)))
        self._computeBBoxes()

    def recalculateBBoxes(self, fn) : 
        for n in self.nodes : 
            self.nodes[n]['bbox'] = fn(self.nodes[n]['bbox'])

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
        normalize(self)

    def _computeBBoxes (self) : 
        paths = [p.path for p in cachedPaths(self.doc)]
        for n in self.nodes : 
            ps = self.nodes[n]['pathSet']
            relPaths = [paths[i] for i in ps]
            bbox = union(map(pathBBox, relPaths))
            nx.set_node_attributes(self, {n: bbox}, 'bbox')

    def _setBBox (self, n) : 
        if 'bbox' in self.nodes[n] : 
            return self.nodes[n]['bbox']
        childBBoxes = [self._setBBox(_) for _ in self.neighbors(n)]
        bbox = union(childBBoxes)
        nx.set_node_attributes(self, {n: bbox}, 'bbox')
        return bbox
        
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
        leafBoxes = [
            *[self.nodes[l]['bbox'] for l in leaves(self)],
            *[that.nodes[l]['bbox'] for l in leaves(that)]
        ]
        for l, bbox in zip(leaves(dataUnion), leafBoxes) : 
            nx.set_node_attributes(dataUnion, {l: bbox}, 'bbox')
        dataUnion._setBBox(newRoot)
        # normalize into a suitable viewbox
        normalize(dataUnion)
        return dataUnion


