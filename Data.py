import sys
import numpy as np
import torch
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

def str2list (s) :
    l = re.split('\[|\]|, ', s)
    l = l[1:-1]
    l = [int(i) for i in l]
    return l

def createTrees (fileTuple) : 
    """
    GRAPH_CLUSTER_ALGO is a dictionary
    containing graph partitioning algorithms.

    Use those algorithms to create trees.

    Helper function while calling parallelize.

    Parameters
    ----------
    fileTuple : tuple
        (Path to file)
    """
    fileName, = fileTuple
    G = relationshipGraph(fileName, RELATION_FUNCTIONS)
    treesAndPrefixes = []
    for key, val in GRAPH_CLUSTER_ALGO.items():
        tree = graphCluster(G, val, svg.Document(fileName), DESC_FUNCTIONS)
        treesAndPrefixes.append((tree, key))
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
            self.root = findRoot(tree)
            self.restoreFile = restoreFile
            self.nPaths = len(self.root.pathSet)

    def setTree(self, tree) :
        self.tree = tree
        self.root = findRoot(tree)
        self.nPaths = len(self.tree.nodes[root]['pathSet'])

    def merge(self, thatTree): 
        # TODO Broken Stuff ahead
        """
        BROKEN
        Merge this tree with thatTree.
        
        This is a helper function used while 
        creating the hierarchy at test time.
        
        A new root node is created with the
        current tree as the left subtree and
        thatTree as the right subtree.

        Parameters
        ----------
        thatTree : Tree
            The right subtree for the merger
        """
        pathSet1 = self.root.pathSet
        pathSet2 = thatTree.root.pathSet
        combine = pathSet1 + pathSet2
        self.root = Node(left=self.root, 
                         right=thatTree.root, 
                         node_type=NodeType.MERGE, 
                         pathSet=combine)
        self.nPaths += thatTree.nPaths

    def save (self, out) :
        GraphReadWrite('tree').write((self.tree, self.root), out) 

class GRASSDataset(data.Dataset):
    """
    How things are done in pytorch 
    to handle data.
    """

    def __init__(self, transform=None):
        """ 
        Constructor

        If the jsons are not there, then generate trees 
        using the parameters present in config.

        Else, use the jsons as the tree dataset.
        """
        files = os.listdir(SAVE_TREES_DIR)

        if len(files) == 0 :
            print("Making trees from scratch!")
            parallelize(
                [DATA_DIRECTORY], 
                SAVE_TREES_DIR, 
                createTrees,
                treeWriter
            )

        files = os.listdir(SAVE_TREES_DIR)
        self.trees = [
            Tree((osp.join(SAVE_TREES_DIR,f))) for f in files
        ]

    def __getitem__(self, index):
        return self.trees[index]

    def __len__(self):
        return len(self.trees)
