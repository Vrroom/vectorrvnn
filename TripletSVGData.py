import svgpathtools as svg
from skimage import transform
import networkx as nx
from raster import *
import relationshipGraph
from relationshipGraph import *
from descriptor import relbb, equiDistantSamples, bb
from functools import reduce
import numpy as np
from graphOps import contractGraph
from treeOps import *
from torchvision import transforms as T
import torch
from svgIO import getTreeStructureFromSVG
from graphIO import GraphReadWrite
from torchUtils import imageForResnet
from itertools import product
from more_itertools import unzip
from functools import lru_cache
import os
import os.path as osp
from osTools import *
import matplotlib.pyplot as plt

def isDegenerateBBox (box) :
    _, _, h, w = box
    return h == 0 and w == 0

class TripletSVGData (nx.DiGraph) : 

    def __init__ (self, svgFile, pickle) : 
        """
        Constructor.

        Role of this object is to aggregate all 
        data items for this svgFile, given the 
        configuration and provide a set of
        methods to conveniently operate recursive 
        networks over them. 
        """
        # The pathSet attribute also has indices which 
        # are the same as how the svgpathtools library
        # orders the paths.
        super(TripletSVGData, self).__init__(nx.read_gpickle(pickle))
        setSubtreeSizes(self)
        setNodeDepths(self)
        setNodeBottomDepths(self)
        self.svgFile = svgFile
        self.doc = svg.Document(svgFile)
        # The nodes in the graph are indexed according
        # to the order they come up in the list.
        self.doc.normalize_viewbox()
        docViewBox = self.doc.get_viewbox()
        paths = cachedFlattenPaths(self.doc)
        paths = [p for i, p in enumerate(paths) if not isDegenerateBBox(relbb(self.doc, i))]
        self.nPaths = len(paths)
        self.pathViewBoxes = [bb(self.doc, i) for i, p in enumerate(paths)]
        with open(svgFile) as fd : 
            self.svg = fd.read()
        for r in [r for r in self.nodes if self.in_degree(r) == 0] : 
            self._computeBBoxes(r)
        pathIdx = list(range(len(paths)))
        self.bigImage = SVGSubset2NumpyImage2(self.doc, pathIdx, 300, 300, alpha=True)
        self.bigImage = np.pad(self.bigImage, ((150, 150), (150, 150), (0, 0)), mode='constant')
        self.pathRasters = []
        for i, p in enumerate(paths) : 
            self.pathRasters.append(SVGSubset2NumpyImage2(self.doc, (i,), 64, 64, alpha=True))
        self._pathSet2Tuple()
        self._computeNodeImages()

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
    
    def mkdir (self, dname) :
        if not osp.exists(dname) : 
            os.mkdir(dname)

    def write (self, root) : 
        dname = osp.join(root, getBaseName(self.svgFile))
        self.mkdir(dname)
        cropDir = osp.join(dname, 'crop')
        wholeDir = osp.join(dname, 'whole')
        self.mkdir(cropDir)
        self.mkdir(wholeDir)
        for n in self.nodes : 
            fname1 = osp.join(cropDir, f'{n}.png')
            fname2 = osp.join(wholeDir, f'{n}.png')
            plt.imsave(fname1, self.nodes[n]['crop'])
            plt.imsave(fname2, self.nodes[n]['whole'])
        treeCopy = nx.DiGraph()
        treeCopy.update(self)
        for n in treeCopy.nodes : 
            del treeCopy.nodes[n]['crop']
            del treeCopy.nodes[n]['whole']
        nx.write_gpickle(treeCopy, osp.join(dname, 'tree.pkl'))
