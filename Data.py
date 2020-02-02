import sys
import numpy as np
import torch
from torch.utils import data
from scipy.io import loadmat
from enum import Enum
import os
import graphOps
import json
import networkx as nx
from networkx.algorithms import bipartite, matching
import svgpathtools as svg
from Utilities import *
import re

def relationshipGraph (svgFile) :
    """
    Take the svg, find graphs
    for all relationship types
    and combine them into one single
    graph for clustering.

    Parameters
    ----------
    svgFile : str
        path to svg file.
    """
    # List of all relationship computers
    relFunctions = [
        adjGraph,
        symGraph
    ]
    graphs = map(lambda x : graphFromSvg(svgFile, x), relFunctions)
    graph = None
    for g in graphs :
        if graph is None: 
            graph = g
        else :
            graph = nx.compose(graph, g)
    return graph

def str2list (s) :
    l = re.split('\[|\]|, ', s)
    l = l[1:-1]
    l = [int(i) for i in l]
    return l

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

    def __init__(self, treeDir, transform=None):
        """ 
        Constructor

        The treeDir contains a bunch of jsons
        representing the tree structure along with 
        path descriptors for each data point. These
        jsons are read and converted into Tree objects.

        Parameters
        ----------
        treeDir : str
            The address of the directory
        """
        self.trees = [Tree((os.path.join(treeDir,f))) for f in os.listdir(treeDir)]

    def __getitem__(self, index):
        return self.trees[index]

    def __len__(self):
        return len(self.trees)
