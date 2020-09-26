import svgpathtools as svg
import networkx as nx
from svgIO import getTreeStructureFromSVG
from raster import SVGtoNumpyImage
import relationshipGraph
from relationshipGraph import *
from descriptor import relbb, equiDistantSamples
from functools import reduce
import numpy as np
from graphOps import contractGraph
from treeOps import findRoot
from torchvision import transforms as T
import torch

class SVGData (nx.DiGraph) : 

    def __init__ (self, svgFile, graph, samples) : 
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
        super(SVGData, self).__init__(getTreeStructureFromSVG(svgFile))
        self.svgFile = svgFile
        self.image = SVGtoNumpyImage(svgFile, H=224, W=224)
        graphFn = getattr(relationshipGraph, graph)
        doc = svg.Document(svgFile)
        paths = doc.flatten_all_paths()
        docViewBox = doc.get_viewbox()
        self.pathViewBoxes = [relbb(p.path, docViewBox) for p in paths]
        # The nodes in the graph are indexed according
        # to the order they come up in the list.
        self.graph = graphFn(paths, vbox=docViewBox)
        nSamples = samples
        self.descriptors = [equiDistantSamples(p.path, docViewBox, nSamples=nSamples) for p in paths]

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
        subgraph = self.graph.subgraph(pathSet).copy()
        h = reduce(lambda g, ps: contractGraph(g, ps), childPathSets, subgraph)
        mapping = dict(map(reversed, enumerate(childPathSets)))
        h = nx.relabel_nodes(h, mapping)
        edges = np.array(h.edges).T
        return torch.from_numpy(edges)

    def image2tensor (self, cuda=False) :
        """
        Since the first pass of the rasterized 
        graphic is through the resnet, we 
        need to normalize it.
        """
        self.image = torch.from_numpy(self.image)
        self.image = self.image.permute(2, 0, 1)
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.image = normalizer(self.image)
        self.image = self.image.unsqueeze(0)
        if cuda :
            self.image = self.image.cuda()
    
    def descriptor2tensor (self, cuda=False) :
        self.descriptors = torch.tensor(self.descriptors)
        if cuda : 
            self.descriptors = self.descriptors.cuda()

    def bbox2tensor (self, cuda=False) :  
        self.pathViewBoxes = torch.tensor(self.pathViewBoxes)
        if cuda : 
            self.pathViewBoxes = self.pathViewBoxes.cuda()

    def toTensor(self, cuda=False) : 
        self.image2tensor(cuda)
        self.descriptor2tensor(cuda)
        self.bbox2tensor(cuda)

if __name__ == '__main__' : 
    data = SVGData('/Users/amaltaas/BTP/vectorrvnn/PartNetSubset/Train/10007.svg', "adjGraph", 10)
    print(data.edgeIndicesAtLevel(findRoot(data)))
