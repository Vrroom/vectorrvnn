from copy import deepcopy
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
import Constants as C
import pydiffvg

def normalizeBBox (box) : 
    x, y, h, w = box
    d = max(h, w)
    return [x - (d - h) / 2, y - (d - w)/ 2, d, d]

def render(canvas_width, canvas_height, shapes, shape_groups):
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None,
                 *scene_args)
    return img

class TripletSVGData (nx.DiGraph) : 

    def preprocessTree (self) : 
        setSubtreeSizes(self)
        setNodeDepths(self)
        setNodeBottomDepths(self)
        self._pathSet2Tuple()

    def preprocessGraphic (self, svgFile) :
        self.svgFile = svgFile
        doc = svg.Document(svgFile)
        docViewBox = normalizeBBox(drawingBBox(doc))
        scale = C.raster_size / docViewBox[-1]
        docViewBox = [_ * scale for _ in docViewBox]
        gstr = f'<g transform="translate({-docViewBox[0]} {-docViewBox[1]}) scale({scale})"></g>'
        groupElt = ET.fromstring(gstr)
        doc.set_viewbox(' '.join(map(str, [0, 0, *docViewBox[-2:]])))
        rootCpy = deepcopy(doc.tree.getroot())
        childrenCpy = list(rootCpy)
        groupElt.extend(childrenCpy)
        root = doc.tree.getroot()
        while len(root) != 0 : 
            for child in root : 
                root.remove(child)
        root.append(groupElt)
        doc.tree._setroot(root)
        self.doc = doc
        self.svg = ET.tostring(self.doc.tree.getroot()).decode()
        paths = cachedFlattenPaths(self.doc)
        paths = [p for i, p in enumerate(paths) if not isDegenerateBBox(relbb(self.doc, i))]
        self.nPaths = len(paths)
        self.pathViewBoxes = [bb(self.doc, i) for i, p in enumerate(paths)]
        # for r in [r for r in self.nodes if self.in_degree(r) == 0] : 
        #     self._computeBBoxes(r)

    def preprocessRasters (self) : 
        pass
        
    def __init__ (self, svgFile, pickle) : 
        """
        Constructor.
        """
        super(TripletSVGData, self).__init__(nx.read_gpickle(pickle))
        self.preprocessTree()
        self.preprocessGraphic(svgFile)
        self.preprocessRasters()

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
