import sys
from torchvision import transforms as T
from copy import deepcopy
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from functools import reduce, partial
import numpy as np
import torch
import string
import more_itertools
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import os
import os.path as osp
import json
import networkx as nx
import svgpathtools as svg
from Utilities import *
import Utilities
import re
import logging
import torch_geometric.data as tgd

class Saveable () : 

    def save (self, savePath) : 
        with open(savePath, 'wb') as fd :
            pickle.dump(self, fd)

class Descriptor (Saveable) : 
    """
    Pre-processed descriptors.
    """
    def __init__ (self, svgDir, descFunction, model) : 
        self.svgDir = svgDir
        self.svgFiles = listdir(svgDir) 
        paths = map (lambda x : svg.Document(x).flatten_all_paths(), self.svgFiles)
        vboxes = map(lambda x : svg.Document(x).get_viewbox(), self.svgFiles)
        allPathsDescFn = AllPathDescriptorFunction(descFunction, model)
        self.descriptors = [allPathsDescFn(p, b) for p, b in zip(paths, vboxes)]

    def __getitem__ (self, i) : 
        return self.descriptors[i]

    def __len__ (self) : 
        return len(self.svgFiles)

    def __or__ (self, that) :
        """
        Concatenation of descriptors 
        """
        assert self.svgDir == that.svgDir 
        assert len(self) == len(that) 
        newDesc = copy.deepcopy(self)
        for i in range(len(newDesc)) : 
            thatDesc = that.descriptors[i] 
            newDesc.descriptors[i] = np.hstack([newDesc.descriptors[i], thatDesc])
        return newDesc

def str2list (s) :
    l = re.split('\[|\]|, ', s)
    l = l[1:-1]
    l = [int(i) for i in l]
    return l

def treeWriter(treesAndPrefixes, path) :
    """
    Wrap graph writing operation
    into a top level function. 

    Helper function while calling
    parallelize.

    Parameters
    ----------
    treesAndPrefixes : list 
        List of trees along with a 
        prefix indicating which graph 
        partitioning algorithm was 
        used to make that tree.
    path : str
        Where to save these trees.
    """
    for rootedTree, prefix in treesAndPrefixes :
        name, ext = osp.splitext(path)
        name = name + '_' + prefix
        path_ = name + '.' + ext
        GraphReadWrite('tree').write(rootedTree, path_)

class NodeType(Enum):
    """ 
    Since we created the tree dataset through
    the recursive graph partitioning algorithm,
    we have only two types of nodes.
    """
    PATH  = 0  # path node
    MERGE = 1  # merge node

class Tree(object):
    """ 
    Tree representing the RvNN structure for
    a datapoint. 
    """

    @classmethod
    def fromFile(restoreFile) : 
        tree = GraphReadWrite('tree').read(restoreFile)
        return Tree(tree)

    def __init__(self, tree) :
        """ 
        Constructor

        Wrap a nx.DiGraph and provide relevant 
        functions to manipulate it.

        Parameters
        ----------
        tree : nx.DiGraph
            RvNN's topology.
        """
        self.tree = tree
        self.root = Utilities.findRoot(self.tree)
        self.nPaths = len(tree.nodes[self.root]['pathSet'])
        self.restoreFile = None
        self.rootCode = None
        self.loss = None
        self.descriptors = None

    def __lt__ (self, other) : 
        return id(self) < id(other)
    
    def addDescriptors (self, descriptors) : 
        """
        Put descriptors at the leaf nodes.
        """
        self.descriptors = descriptors

    def toTensor (self, cuda=False) : 
        if isinstance(self.descriptors, np.ndarray) : 
            if cuda : 
                self.descriptors = torch.from_numpy(self.descriptors).float().cuda()
            else : 
                self.descriptors = torch.from_numpy(self.descriptors).float()

    def toNumpy (self) :
        if torch.is_tensor(self.descriptors) : 
            self.descriptors = self.descriptors.numpy()

    def path (self, node) : 
        assert len(self.tree.nodes[node]['pathSet']) == 1
        pathId = self.tree.nodes[node]['pathSet'][0]
        return self.descriptors[pathId].reshape((1, -1))

    def setSVGAttributes (self, paths, vb) :
        def getter (T, n, _) :
            lst = self.tree.nodes[n]['pathSet']
            self.tree.nodes[n]['svg'] = getSubsetSvg(paths, lst, vb)
        treeApplyChildrenFirst(self.tree, self.root, getter)

    def save (self, out) :
        GraphReadWrite('tree').write((self.tree, self.root), out) 

class TreeCreator () :
    """
    Create trees by using the clustering algos 
    present in graphClusterAlgos on the graph created
    from the SVG using relationFunctions. 

    mp.Pool can't handle lambda functions. 
    Something about them being unpickleable. 

    As a result, I have to provide context using
    this class. Although now I realized that I 
    could use partial!
    """
    def __init__ (self, graphClusterAlgo, relationFunctions) :
        """
        Constructor. 

        Parameters
        ----------
        graphClusterAlgo : function
            One graphClusteringAlgorithm
        relationFunctions : list
            They detect whether an edge exists between
            a pair of paths.
        """
        self.graphClusterAlgo = graphClusterAlgo
        self.relationFunctions = relationFunctions

    def __call__ (self, fileName) : 
        """
        Make this class callable.

        Parameters
        ----------
        fileName : str
            Path to the input SVG file.
        """
        G = relationshipGraph(fileName, self.relationFunctions)
        dendogram, _ = graphCluster(G, self.graphClusterAlgo, svg.Document(fileName))
        tree = Tree(dendogram)
        return tree, self.graphClusterAlgo.__name__

class TreesData (data.Dataset, Saveable) : 
    """
    Pre-processed trees from clustering 
    algorithm.
    """
    def __init__ (self, svgDir, descFunction, model) : 
        self.svgDir = svgDir
        self.descFunction = descFunction
        self.svgFiles = listdir(svgDir) 
        self.tensor = False
        self.model = model
        with ProcessPoolExecutor() as executor : 
            self.trees = list(executor.map(getTreeStructureFromSVG, self.svgFiles, chunksize=4))
            self.rasterImages = list(executor.map(partial(SVGtoNumpyImage, H=224, W=224), self.svgFiles, chunksize=4))
        self.descriptors = Descriptor(svgDir, descFunction, model)
        for t, d in zip(self.trees, self.descriptors) :
            t.addDescriptors(d)

    def dataFromDocument (self, filename) : 
        doc = svg.Document(filename)
        vb = doc.get_viewbox()
        paths = doc.flatten_all_paths()
        edges = Pose.boneConnectivity
        descriptors = [self.descFunction(p.path, vb) for p in paths]
        descriptors = torch.tensor(descriptors)
        pathTypes = [p.element.attrib['id'] for p in paths]
        pathTypeIdx = dict(map(reversed, enumerate(pathTypes)))
        targets = torch.tensor([pathTypeIdx[t] for t in pathTypes]).long()
        edges = [(pathTypeIdx[a], pathTypeIdx[b]) for a, b in edges]
        edges = torch.t(torch.tensor(edges).long())
        return tgd.Data(x=descriptors, edge_index=edges, y=targets)

    def imagesToTensor (self, cuda) :
        self.rasterImages = list(map(torch.from_numpy, self.rasterImages))
        self.rasterImages = [img.permute(2, 0, 1) for img in self.rasterImages]
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.rasterImages = list(map(normalizer, self.rasterImages))
        self.rasterImages = [img.unsqueeze(0) for img in self.rasterImages]
        if cuda :
            self.rasterImages = [img.cuda() for img in self.rasterImages]

    def __getitem__ (self, index) :
        tree = self.trees[index]
        image = self.rasterImages[index]
        return tree, image

    def __len__ (self) : 
        return len(self.svgFiles)

    def treesToTensor (self, cuda) : 
        for tree in self.trees : 
            tree.toTensor(cuda)

class DataHandler (Saveable) :
    """
    There will be configurations that 
    will be common across experiments. So we need 
    to cache whatever was common across them. 
    """

    def __init__ (self, svgDir) : 
        self.svgDir = svgDir
        self.svgFiles = listdir(svgDir)
        self.treeCache = dict() 
        self.descCache = dict() 
        with ProcessPoolExecutor() as executor : 
            self.groundTruth = list(executor.map(getTreeStructureFromSVG, self.svgFiles, chunksize=4))

    def getTrees (self, descFunctions, cuda, model) :
        treeList = []
        key = str(descFunctions)
        if key not in self.treeCache :
            self.treeCache[key] = TreesData(self.svgDir, ComposeAdd(descFunctions), model)
            self.treeCache[key].imagesToTensor(cuda)
        treeList.append(self.treeCache[key])
        return reduce(lambda x, y : x + y, treeList)

    def getDataset (self, config, cuda) : 
        functionGetter = lambda x : getattr(Utilities, x) 
        descFunctions = list(map(functionGetter, config['desc_functions']))
        model = torch.load(config['model_path'])
        model.eval()
        trees = self.getTrees(descFunctions, cuda, model)
        trees.treesToTensor(cuda)
        return trees
