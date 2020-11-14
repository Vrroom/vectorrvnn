import svgpathtools as svg
import networkx as nx
from raster import SVGtoNumpyImage
import relationshipGraph
from relationshipGraph import *
from descriptor import relbb, equiDistantSamples, pathAttr
from functools import reduce
import numpy as np
from graphOps import contractGraph
from treeOps import findRoot, treeApplyChildrenFirst, lca, setNodeDepths, maxDepth
from torchvision import transforms as T
import torch
from svgIO import getTreeStructureFromSVG
from graphIO import GraphReadWrite

def isDegenerateBBox (box) : 
    _, _, h, w = box
    return h == 0 and w == 0 

class SVGData (nx.DiGraph) : 

    def __init__ (self, svgFile, treeJson, graph, samples, useColor=True) : 
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
        super(SVGData, self).__init__(GraphReadWrite('tree').read(treeJson))
        self.root = findRoot(self)
        self.svgFile = svgFile
        # self.image = SVGtoNumpyImage(svgFile, H=224, W=224)
        self.image = np.zeros((224, 224, 3))
        graphFn = getattr(relationshipGraph, graph)
        doc = svg.Document(svgFile)
        docViewBox = doc.get_viewbox()
        paths = doc.flatten_all_paths()
        paths = [p for p in paths if not isDegenerateBBox(relbb(p.path, docViewBox))]
        #TODO : Do something about color as well.
        self.nPaths = len(paths)
        self.pathViewBoxes = [relbb(p.path, docViewBox) for p in paths]
        # The nodes in the graph are indexed according
        # to the order they come up in the list.
        self.adjgraph = graphFn(paths, vbox=docViewBox)
        nSamples = samples
        if useColor : 
            self.descriptors = [[*equiDistantSamples(p.path, docViewBox, nSamples=nSamples), pathAttr(p, docViewBox)] for p in paths]
        else : 
            self.descriptors = [equiDistantSamples(p.path, docViewBox, nSamples=nSamples) for p in paths]
        self._computeBBoxes(self.root)
        self._pathSet2Tuple()
        self._computeLCAMatrix()

    def _computeLCAMatrix(self) : 
        self.lcaMatrix = np.zeros((self.nPaths, self.nPaths))
        self.maxDepth = maxDepth(self)
        setNodeDepths(self)
        for i in range(self.nPaths) : 
            for j in range(self.nPaths) : 
                self.lcaMatrix[i, j] = self.nodes[lca(self, i, j)]['depth'] / self.maxDepth

    def _nodeId2PathId (self, n) : 
        assert self.out_degree(n) == 0, "Function called with internal node"
        return self.nodes[n]['pathSet'][0]
    
    def _path (self, n) : 
        return self.descriptors[self._nodeId2PathId(n)]

    def _color (self, n) : 
        return self.color[self._nodeId2PathId(n)]

    def _bbox (self, n) : 
        return self.pathViewBoxes[self._nodeId2PathId(n)]

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
            xm, ym = boxes[:, 0].min(), boxes[:, 2].min()
            xM, yM = (xm + boxes[:, 2]).max(), (ym + boxes[:, 3]).max()
            nx.set_node_attributes(self, {node: [xm, ym, xM - xm, yM - ym]}, 'bbox')

    def edgeIndicesAtLevel (self, node) : 
        """
        Assuming that the order of the node 
        features is same as the order of 
        the children of this node, output a
        numpy array [2, num_edges] which represents
        the subgraph induced on the children of this
        node.
        """
        pathSet = self.nodes[node]['pathSet']
        childPathSets = [tuple(self.nodes[n]['pathSet']) for n in self.neighbors(node)]
        subgraph = self.adjgraph.subgraph(pathSet).copy()
        h = reduce(lambda g, ps: contractGraph(g, ps), childPathSets, subgraph)
        mapping = dict(map(reversed, enumerate(childPathSets)))
        h = nx.relabel_nodes(h, mapping)
        edges = np.array(h.edges).T.reshape((2, -1))
        return torch.from_numpy(edges).long()

    def edge_index (self, cuda=False) :
        edges = torch.from_numpy(np.array(self.adjgraph.edges).T).long()
        edges = edges.view((2, -1))
        return edges

    def lcaMatrix2tensor(self, cuda=False) : 
        if isinstance(self.lcaMatrix, torch.Tensor) : 
            return
        self.lcaMatrix = torch.from_numpy(self.lcaMatrix).float()

    def image2tensor (self, cuda=False) :
        """
        Since the first pass of the rasterized 
        graphic is through the resnet, we 
        need to normalize it.
        """
        if isinstance(self.image, torch.Tensor) : 
            return
        self.image = torch.from_numpy(self.image).float()
        self.image = self.image.permute(2, 0, 1)
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.image = normalizer(self.image)
        self.image = self.image.unsqueeze(0)
        if cuda :
            self.image = self.image.cuda()
    
    def descriptor2tensor (self, cuda=False) :
        if isinstance(self.descriptors, torch.Tensor) : 
            return
        self.descriptors = torch.tensor(self.descriptors)
        if cuda : 
            self.descriptors = self.descriptors.cuda()

    def bbox2tensor (self, cuda=False) :  
        if isinstance(self.pathViewBoxes, torch.Tensor) : 
            return
        self.pathViewBoxes = torch.tensor(self.pathViewBoxes).float()
        for n in self.nodes : 
            self.nodes[n]['bbox'] = torch.tensor(self.nodes[n]['bbox']).float()
        if cuda : 
            self.pathViewBoxes = self.pathViewBoxes.cuda()
            for n in self.nodes : 
                self.nodes[n]['bbox'] = self.nodes[n]['bbox'].cuda()

    def toTensor(self, cuda=False) : 
        self.image2tensor(cuda)
        self.descriptor2tensor(cuda)
        self.bbox2tensor(cuda)
        self.lcaMatrix2tensor(cuda)

if __name__ == '__main__' : 
    data = SVGData('/Users/amaltaas/BTP/vectorrvnn/PartNetSubset/Train/10007.svg', "adjGraph", 10)
    print(data.edgeIndicesAtLevel(findRoot(data)))