import svgpathtools as svg
import matplotlib.pyplot as plt
from skimage import transform
import networkx as nx
from raster import *
import relationshipGraph
from relationshipGraph import *
from descriptor import relbb, equiDistantSamples, pathAttr, bb
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

def isDegenerateBBox (box) : 
    _, _, h, w = box
    return h == 0 and w == 0 

class TripletSVGData (nx.DiGraph) : 

    def __init__ (self, svgFile, pickle, graph, samples, useColor=True) : 
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
        self.doc = svg.Document(svgFile)
        self.doc.normalize_viewbox()
        docViewBox = self.doc.get_viewbox()
        paths = self.doc.flatten_all_paths()
        paths = [p for p in paths if not isDegenerateBBox(relbb(p.path, docViewBox))]
        self.nPaths = len(paths)
        self.pathViewBoxes = [bb(p.path) for p in paths]
        for r in [r for r in self.nodes if self.in_degree(r) == 0] : 
            self._computeBBoxes(r)
        self.bigImage = SVGtoNumpyImage(svgFile, 300, 300, alpha=True)
        self.bigImage = np.pad(self.bigImage, ((150, 150), (150, 150), (0, 0)), mode='constant')
        self.pathRasters = []
        for i, p in enumerate(paths) : 
            self.pathRasters.append(SVGSubset2NumpyImage2(self.doc, (i,) , 32, 32, alpha=True))
        self.image = SVGtoNumpyImage(svgFile, 32, 32, alpha=True)
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
    
    def pathSetCrop (self, n) :
        xm, ym, h, w = self.nodes[n]['bbox']
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
        return transform.resize(im, (32, 32))

    def _computeNodeImages (self) : 
        for n in self.nodes : 
            ps  = self.nodes[n]['pathSet']
            self.nodes[n]['crop'] = self.pathSetCrop(n)
            # self.nodes[n]['whole'] = SVGSubset2NumpyImage2(self.doc, ps, 32, 32, alpha=True)
            plt.imshow(self.nodes[n]['crop'])
            plt.savefig(f'./pngs/{n}.png')
            plt.close()
