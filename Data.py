import sys
import pickle
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
    this class.
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
            dendogram = graphCluster(G, algo, 
                    svg.Document(fileName), self.descFunctions)
            tree = Tree()
            tree.setTree(*dendogram)
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
            
            self.tensorify()
            self.nPaths = len(self.tree.nodes[self.root]['pathSet'])
    
    def tensorify (self) : 
        for n in self.tree.nodes :
            if 'desc' in self.tree.nodes[n] : 
                tensorified = torch.tensor(self.tree.nodes[n]['desc']).cuda()
                self.tree.nodes[n]['desc'] = tensorified

    def untensorify (self) :
        for n in self.tree.nodes :
            if 'desc' in self.tree.nodes[n] : 
                self.tree.nodes[n]['desc'] = self.tree.nodes[n]['desc'].tolist()

    def setTree(self, tree, root) :
        """
        To be used when a restore file is
        not specified. The tree is created 
        from a nx.DiGraph

        Parameters
        ----------
        tree : nx.DiGraph
        """
        self.tree = tree
        self.root = root
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

        with mp.Pool(mp.cpu_count()) as p : 
            self.groundTruth = p.map(getTreeStructureFromSVG, self.svgFiles)

        if makeTrees : 
            creator = TreeCreator(**kwargs)
            with mp.Pool(mp.cpu_count()) as p : 
                self.trees = p.map(creator, self.svgFiles)

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
        for svgFile, trees in zip(self.svgFiles, self.trees) : 
            head, = osp.split(svgFile)
            name, _ = osp.splitext(head)
            treeWriter(trees, osp.join(savePath, name + '.json'))
