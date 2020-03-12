import sys
import multiprocessing as mp
import numpy as np
import torch
import string
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import os
import os.path as osp
import json
import networkx as nx
import svgpathtools as svg
from Utilities import *
from Config import *
import re
import logging

def str2list (s) :
    l = re.split('\[|\]|, ', s)
    l = l[1:-1]
    l = [int(i) for i in l]
    return l

def createTrees (fileTuple, graphClusterAlgo=None, 
        relationFunctions=None, descFunctions=None) : 
    """
    Create trees by using the clustering algos 
    present in graphClusterAlgo on the graph created
    from the SVG using relationFunctions. 

    Helper function while calling parallelize.

    Parameters
    ----------
    fileTuple : tuple
        (Path to file)
    graphClusterAlgo : None or list
        Collection of graph clustering algorithms
        to build dendograms.
    relationFunctions : list
        They detect whether an edge exists between
        a pair of paths.
    descFunctions : list
        Function used to compute the path 
        descriptors.
    """
    fileName, = fileTuple
    G = relationshipGraph(fileName, relationFunctions)
    treesAndPrefixes = []
    for algo in graphClusterAlgo:
        tree = graphCluster(G, algo, svg.Document(fileName), descFunctions)
        treesAndPrefixes.append((tree, algo.__name__))
    return treesAndPrefixes

def treeWriter(treesAndPrefixes, name) :
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
    name : str
        Name of the data point to 
        save by.
    """
    for rootedTree, prefix in treesAndPrefixes :
        fileName = name + '_' + prefix + '.json'
        GraphReadWrite('tree').write(rootedTree, fileName)

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

    def __init__(self, restoreFile=None) :
        """ 
        Constructor

        Takes a json file and constructs a tree object.
        the json file represents a tree in networkx's
        tree graph format. The tree is retrieved 
        from the json and tensorified.

        If the restoreFile is not given, then you
        have to give a nx.DiGraph tree.

        Parameters
        ----------
        restoreFile : str
            File address    
        """

        self.nPaths = 0
        if not restoreFile is None :
            self.tree = GraphReadWrite('tree').read(restoreFile)
            self.root = findRoot(self.tree)
            self.restoreFile = restoreFile
            
            for n in self.tree.nodes :
                if 'desc' in self.tree.nodes[n] :
                    tensorified = torch.tensor(self.tree.nodes[n]['desc']).cuda()
                    self.tree.nodes[n]['desc'] = tensorified

            self.nPaths = len(self.tree.nodes[self.root]['pathSet'])

    def setTree(self, tree) :
        """
        To be used when a restore file is
        not specified. The tree is created 
        from a nx.DiGraph

        Parameters
        ----------
        tree : nx.DiGraph
        """
        self.tree = tree
        self.root = findRoot(tree)
        self.nPaths = len(self.tree.nodes[self.root]['pathSet'])

    def merge(self, thoseTrees, paths, vbox): 
        """
        Merge this tree with a list of trees.
        
        This is a helper function used while 
        creating the hierarchy at test time.
        
        The new node's label is some random 
        string.

        Parameters
        ----------
        thoseTrees : list
            A list of subtrees to be combined with
        paths : list
            List of paths to create the svg string
            for the merge
        vbox : list
            Bounding Box. To be used to create the
            svg string.
        """
        newRoot = randomString(10)

        nxThoseTrees = list(map(lambda x : x.tree, thoseTrees))
        self.tree = nx.compose_all([self.tree, *nxThoseTrees])
        self.tree.add_node(newRoot)

        newEdges = list(map(lambda x : (newRoot, x.root), thoseTrees))
        self.tree.add_edge(newRoot, self.root)
        self.tree.add_edges_from(newEdges)

        self.root = newRoot

        self.tree.nodes[self.root]['pathSet'] = list(more_itertools.collapse(
            [
                self.tree.nodes[i]['pathSet'] 
                for i 
                in self.tree.neighbors(self.root)
            ]
        ))

        self.tree.nodes[self.root]['svg'] = getSubsetSvg(
            paths,
            self.tree.nodes[self.root]['pathSet'],
            vbox
        )

        self.nPaths += sum([t.nPaths for t in thoseTrees])

    def relabel(self) :
        """
        At test time, while merging, we have 
        to pick arbitrary labels for the 
        internal nodes.

        This will fix that and give integer
        labels to all nodes.
        """
        strLabels = filter(lambda x : isinstance(x, str), self.tree.nodes)
        newLabels = range(self.nPaths, self.tree.number_of_nodes())
        mapping = dict(zip(strLabels, newLabels))
        self.tree = nx.relabel_nodes(self.tree, mapping, copy=False) 
        self.root = mapping[self.root]

    def save (self, out) :
        GraphReadWrite('tree').write((self.tree, self.root), out) 

class GRASSDataset(data.Dataset):
    """
    How things are done in pytorch 
    to handle data.
    """

    def __init__(self, svgDir, treesDir=None, transform=None, **kwargs):
        """ 
        Constructor.

        If treesDir is given, use those
        trees in the dataset. Else prepare
        them from scratch using heuristics.

        Also, compute ground truth.
        
        Parameters
        ----------
        svgDir : str
            Path to SVGs.
        treesDir : str or None
            Where to store the trees constructed of
            the heuristics.
        transform : 
            Some function to apply a transformation
            on the dataset.
        """
        self.svgDir = svgDir
        self.treesDir = treesDir
        self.transform = transform

        self.svgFiles = listdir(svgDir)

        with mp.Pool(mp.cpu_count()) as p : 
            self.groundTruth = p.map(getTreeStructureFromSVG, self.svgFiles)

        if treesDir is not None : 
            files = listdir(treesDir)

            if len(files) == 0 :
                logging.info("Making trees from scratch!")
                parallelize(
                    [svgDir], 
                    treesDir, 
                    createTrees,
                    treeWriter,
                    **kwargs
                )

            files = listdir(treesDir)
            self.trees = [Tree(f) for f in files]

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
        if self.treesDir is None : 
            return self.svgFiles[index], self.groundTruth[index]
        else :
            return self.svgFiles[index], self.groundTruth[index], self.trees[index]

    def __len__(self):
        return len(self.svgFiles)
