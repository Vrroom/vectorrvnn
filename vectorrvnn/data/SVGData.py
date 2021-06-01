from functools import reduce, lru_cache
import svgpathtools as svg
from skimage import transform
import networkx as nx
import numpy as np
from vectorrvnn.geometry import *
from vectorrvnn.utils import *

class SVGData (nx.DiGraph) : 

    def preprocessTree (self) : 
        setSubtreeSizes(self)
        setNodeDepths(self)
        setNodeBottomDepths(self)
        self._pathSet2Tuple()

    def preprocessGraphic (self, svgFile) :
        self.svgFile = svgFile
        self.doc = svg.Document(svgFile)
        paths = cachedPaths(self.doc)
        paths = [p for i, p in enumerate(paths) 
                if not isDegenerateBBox(bb(self.doc, i))]
        self.nPaths = len(paths)
        self.pathViewBoxes = [bb(self.doc, i) for i, p in enumerate(paths)]
        for r in [r for r in self.nodes if self.in_degree(r) == 0] : 
            self._computeBBoxes(r)

    def __init__ (self, svgFile, treePickle=None, tree=None) : 
        """ Only one of treePickle and tree can be not None """
        assert(treePickle is None or tree is None)
        if treePickle is not None : 
            super(SVGData, self).__init__(nx.read_gpickle(treePickle))
        elif tree is not None : 
            super(SVGData, self).__init__(tree)
        else :
            super(SVGData, self).__init__()
        self.preprocessTree()
        self.preprocessGraphic(svgFile)
        # self.preprocessRasters()

    def _pathSet2Tuple (self) : 
        for n in self.nodes :
            self.nodes[n]['pathSet'] = tuple(self.nodes[n]['pathSet'])

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
    
    @lru_cache(maxsize=128)
    def pathSetCrop (self, ps) :
        paths = cachedFlattenPaths(self.doc)
        boxes = np.array([paths[i].path.bbox() for i in ps])
        xm, xM = boxes[:,0].min(), boxes[:,1].max()
        ym, yM = boxes[:,2].min(), boxes[:,3].max()
        h, w = xM - xm, yM - ym
        docXm, docYm, docH, docW = self.doc.get_viewbox()
        docD = min(docH, docW)
        d = max(h, w)
        eps = docD / 20
        if h > w :
            box = [xm, ym + w/2 - d/2, d, d]
        else : 
            box = [xm + h/2 - d/2, ym, d, d]
        box[0] -= eps
        box[1] -= eps
        box[2] += 2 * eps
        box[3] += 2 * eps
        f = lambda x : int(300 * ((x - docXm) / docH) + 150)
        g = lambda y : int(300 * ((y - docYm) / docW) + 150)
        im = self.bigImage[g(box[1]):g(box[1]+box[3]), f(box[0]):f(box[0]+box[2]), :]
        return transform.resize(im, (64, 64))

    def _computeNodeImages (self) : 
        for n in self.nodes : 
            ps  = self.nodes[n]['pathSet']
            self.nodes[n]['crop'] = self.pathSetCrop(ps)
            self.nodes[n]['whole'] = self.alphaComposite(ps)

    @lru_cache(maxsize=128)
    def alphaComposite (self, pathSet) :
        pathSet = tuple(sorted(pathSet))
        n = len(pathSet)
        if n == 1 :
            return self.pathRasters[pathSet[0]]
        else :
            l = self.alphaComposite(pathSet[:n//2])
            r = self.alphaComposite(pathSet[n//2:])
            cl, cr = l[:, :, :3], r[:, :, :3]
            al, ar = l[:, :, 3:], r[:, :, 3:]
            ao = ar + al * (1 - ar)
            co = (cr * ar + cl * al * (1 - ar))
            co[(ao > 0).squeeze()] /= ao[(ao > 0).squeeze()]
            o = np.concatenate((co, ao), axis=2)
            return o
    
