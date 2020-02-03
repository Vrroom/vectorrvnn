# python3 Test.py <encoder_model_path> <decoder_model_path>
import sys
import math
import os
import torch
import more_itertools
import itertools
from torch import nn
import torch.utils.data
from Data import *
from Model import GRASSEncoder
from Model import GRASSDecoder
import Model
import networkx as nx
import svgpathtools as svg
import numpy as np
from copy import deepcopy 
from Utilities import *
from Config import *

def getModels (encoderPath, decoderPath) :
    """
    Use the given paths to load encoder
    and decoder models.

    Parameters
    ----------
    encoderPath : str
        Path to the saved encoder model.
    decoderPath : str
        Path to the saved decoder model.
    """
    torch.cuda.set_device(GPU)

    if CUDA and torch.cuda.is_available():
        print("Using CUDA on GPU ", GPU)
    else:
        print("Not using CUDA.")

    encoder = torch.load(encoderPath)
    decoder = torch.load(decoderPath)

    if CUDA :
        encoder.cuda()
        decoder.cuda()
        
    return encoder, decoder


def findTree (svgFile, encoder, decoder) :
    """
    Do a local greedy search to find a hierarchy.
    
    Roughly, the algorithm is the follows:
        1) Keep a list of all sub-trees formed so far.
           Initially this is a list of leaf nodes.
        2) Try all possible pairs of trees in the list
           and find the one whose merge results in the
           lowest loss.
        3) Remove the 2 trees that were merged and add
           the merged tree back.
        4) If the list has only one tree left, return
           this tree.

    Parameters
    ----------
    svgFile : str
        Path to svgFile
    encoder : GRASSEncoder
        Encoding model
    decoder : GRASSDecoder
        Decoding model

    """
    doc = svg.Document(svgFile)
    paths = svg.Document(svgFile).flatten_all_paths()
    vb = doc.get_viewbox()

    trees = [] 

    # Populate the candidate list with leaves
    for idx, path in enumerate(paths) :
        descriptors = [f(paths[idx].path, vb) for f in DESC_FUNCTIONS]
        flattened = list(more_itertools.collapse(descriptors))

        nxTree = nx.DiGraph()
        nxTree.add_node(idx)

        nxTree.nodes[idx]['desc'] = torch.tensor(flattened).cuda()
        nxTree.nodes[idx]['pathSet'] = [idx]
        nxTree.nodes[idx]['svg'] = getSubsetSvg(paths, [idx], vb)
        
        tree = Tree()
        tree.setTree(nxTree)
        trees.append(tree)

    # Do local greedy search over the space of 
    # candidate trees.
    while len(trees) > 1 : 
        minI, minJ = -1, -1
        minLoss = math.inf
        bestTree = None
        for i in range(len(trees)) :
            for j in range(len(trees)) :
                if i != j : 
                    treeI = deepcopy(trees[i])
                    treeJ = deepcopy(trees[j])

                    treeI.merge([treeJ], paths, vb)

                    loss = Model.treeLoss(treeI, encoder, decoder).item()
                    
                    if loss < minLoss :
                        minI = min(i, j)
                        minJ = max(i, j)
                        minLoss = loss
                        bestTree = treeI
         
        trees.remove(trees[minI])
        trees.remove(trees[minJ - 1])
        trees.append(bestTree)

    trees[0].relabel()
    return trees[0]

def main () :
    # This operation of generating trees
    # at test time is not parallelizable 
    # because of some shit to do with Cuda
    encoderPath = sys.argv[1]
    decoderPath = sys.argv[2]

    encoder, decoder = getModels(encoderPath, decoderPath)

    for f in os.listdir(TEST_DATA) :
        name = f[:f.rfind('.')]
        fname = osp.join(TEST_DATA, f)
        tree = findTree(fname, encoder, decoder)
        for n in tree.tree.nodes :
            if 'desc' in tree.tree.nodes[n] : 
                tree.tree.nodes[n]['desc'] = tree.tree.nodes[n]['desc'].tolist()
        saveName = osp.join(SAVE_TEST_TREES_DIR, name + '.json')
        GraphReadWrite('tree').write((tree.tree, tree.root), saveName)

if __name__ == "__main__" :
    main()
