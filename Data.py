import sys
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
    def __init__ (self, graphClusterAlgos, relationFunctions, descFunctions) :
        """
        Constructor. 

        Parameters
        ----------
        graphClusterAlgos : list
            Collection of graph clustering algorithms
            to build dendograms.
        relationFunctions : list
            They detect whether an edge exists between
            a pair of paths.
        descFunctions : list
            Function used to compute the path 
            descriptors.
        """
        self.graphClusterAlgos = graphClusterAlgos
        self.relationFunctions = relationFunctions
        self.descFunctions = descFunctions

    def __call__ (self, fileName) : 
        """
        Make this class callable.

        Parameters
        ----------
        fileName : str
            Path to the input SVG file.
        """
        G = relationshipGraph(fileName, self.relationFunctions)
        treesAndPrefixes = []
        for algo in self.graphClusterAlgos:
            dendogram, _ = graphCluster(G, algo, 
                    svg.Document(fileName), self.descFunctions)
            tree = Tree(dendogram)
            tree.tensorify()
            treesAndPrefixes.append((tree, algo.__name__))
        return treesAndPrefixes

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

    def __lt__ (self, other) : 
        return id(self) < id(other)
    
    def addDescriptor (self, descFunctions, paths, vbox) : 
        """
        Put descriptors at the leaf nodes.
        """
        for n in self.tree.nodes : 
            if len(self.tree.nodes[n]['pathSet']) == 1: 
                i = self.tree.nodes[n]['pathSet'][0]
                self.tree.nodes[n]['desc'] = list(more_itertools.collapse(
                    [f(paths[i].path, vbox) for f in descFunctions]
                ))


    def tensorify (self) : 
        for n in self.tree.nodes :
            if 'desc' in self.tree.nodes[n] : 
                tensorified = torch.tensor(self.tree.nodes[n]['desc']).reshape((1, -1))
                self.tree.nodes[n]['desc'] = tensorified

    def untensorify (self) :
        for n in self.tree.nodes :
            if 'desc' in self.tree.nodes[n] : 
                self.tree.nodes[n]['desc'] = self.tree.nodes[n]['desc'].tolist()

    def setSVGAttributes (self, paths, vb) :
        def getter (n, *args) :
            lst = self.tree.nodes[n]['pathSet']
            return getSubsetSvg(paths, lst, vb)
        treeApply(self.tree, self.root, getter)

    def save (self, out) :
        GraphReadWrite('tree').write((self.tree, self.root), out) 

class GRASSDataset(data.Dataset):
    """
    How things are done in pytorch 
    to handle data.
    """

    def __init__(self, svgDir, makeTrees=False, transform=None, **kwargs):
        """ 
        Constructor.

        If you need to make dendograms 
        from scratch, do that.
        Also, compute ground truth.

        Parameters
        ----------
        svgDir : str
            Path to SVGs.
        makeTrees : bool
            Whether to make trees or not.
        transform : 
            Some function to apply a transformation
            on the dataset.
        """
        self.svgDir = svgDir
        self.makeTrees = makeTrees
        self.transform = transform

        self.svgFiles = listdir(svgDir)

        with ProcessPoolExecutor() as executor :
            self.groundTruth = executor.map(getTreeStructureFromSVG, self.svgFiles, chunksize=10)
        
        if makeTrees : 
            with ProcessPoolExecutor() as executor : 
                creator = TreeCreator(**kwargs)
                self.trees = executor.map(creator, self.svgFiles, chunksize=10)

    def __getitem__(self, index):
        """
        The same class can be used depending on
        whether the data is for Training, Validation
        or testing. The datapoint is different 
        depending on what the usage is. 

        For Training, you need to also compute the
        training trees using our heuristic algorithm.

        For Validation and Testing, only the SVG file
        and the ground truth tree constitute the 
        datapoint.
        """
        if not self.makeTrees: 
            return self.svgFiles[index], self.groundTruth[index]
        else :
            return self.svgFiles[index], self.groundTruth[index], self.trees[index]

    def __len__(self):
        return len(self.svgFiles)

    def save (self, savePath) : 
        """
        Save all the training trees in this
        directory.

        Parameters
        ----------
        savePath : str
            Path to be saved to.
        """
        with open(savePath, 'wb') as fd :
            pickle.dump(self, fd)
