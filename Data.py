import sys
from copy import deepcopy
import pickle
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
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
import re
import logging
from functools import partial

def str2list (s) :
    l = re.split('\[|\]|, ', s)
    l = l[1:-1]
    l = [int(i) for i in l]
    return l

class AllPathDescriptorFunction () : 
    """ 
    Given a descriptor function, convert it into
    one that acts on all paths
    """

    def __init__ (self, descFunction) : 
        self.descFunction = descFunction

    def __call__ (self, paths, vbox) : 
        descs = []
        for path in paths : 
            descs.append(self.descFunction(path, vbox))
        return np.vstack(descs)

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
        self.root = findRoot(self.tree)
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
                self.descriptors = torch.from_numpy(self.descriptors).cuda()
            else : 
                self.descriptors = torch.from_numpy(self.descriptors)

    def toNumpy (self) :
        if torch.is_tensor(self.descriptors) : 
            self.descriptors = self.descriptors.numpy()

    def path (self, node) : 
        assert len(self.nodes[node]['pathSet']) == 1
        pathId = self.nodes[node]['pathSet'][0]
        return self.descriptors[pathId]

    def setSVGAttributes (self, paths, vb) :
        def getter (n, *args) :
            lst = self.tree.nodes[n]['pathSet']
            return getSubsetSvg(paths, lst, vb)
        treeApply(self.tree, self.root, getter)

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
    def __init__ (self, svgDir, graphClusteringAlgo, relationFunctions) : 
        self.svgDir = svgDir
        self.transform = transform
        self.svgFiles = listdir(svgDir) 
        creator = TreeCreator(graphClusterAlgo, relationFunctions)
        with ProcessPoolExecutor() as executor : 
            self.trees = executor.map(creator, self.svgFiles, chunksize=10)

    def __getitem__ (self, index) : 
        return self.trees[index]

    def __len__ (self) : 
        return len(self.svgFiles)

    def toTensor (self, cuda) : 
        """
        Whether to use CUDA.
        """
        for tree in self.trees : 
            tree.toTensor(cuda)

class DataHandler () : 
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
            self.groundTruth = executor.map(getTreeStructureFromSVG, self.svgFiles)

    def getDescriptors (self, descFunctions) : 
        descList = []
        for fn in descFunctions : 
            if fn not in self.descCache : 
                self.descCache[fn] = Descriptor(self.svgDir, fn)
            descList.append(self.descCache[fn])
        return reduce(lambda x, y : x | y, descList)

    def getTrees (self, graphClusteringAlgos, relationFunctions) :
        treeList = []
        for algo in graphClusterAlgos : 
            if (algo, relationFunctions) not in self.treeCache :
                self.treeCache[(algo, relationFunctions)] = TreesData(svgDir, algo, relationFunctions)
            treeList.append(self.treeCache[(algo, relationFunctions)])
        return reduce(lambda x, y : x + y, treeList)

    def getDataset (self, config) : 
        functionGetter = lambda x : getattr(Utilities, x) 

        descFunctions = list(map(functionGetter, config['desc_functions']))
        relationFunctions = list(map(functionGetter, config['relation_functions']))
        graphClusterAlgos = list(map(functionGetter, config['graph_cluster_algo']))

        trees = self.getTrees(graphClusterAlgos, relationFunctions)
        descs = self.getDescriptors(descFunctions)
        
        for t, d in zip(trees, descs) : 
            t.addDescriptors(d)

        return trees
            

