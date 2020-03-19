from Data import Tree
import torch
import heapq
from copy import deepcopy
from functools import partial
import svgpathtools as svg
import itertools
import more_itertools
import networkx as nx
import cProfile
import Utilities
from Utilities import *

def findTree (config, svgFile, encoder, decoder) :
    """
    Find the best tree by local greedy search. 

    Parameters
    ----------
    config : dict
    svgFile : str
    encoder : GRASSEncoder
    decoder : GRASSDecoder
    """
    def computeMerges(pair) :
        tl, tr = pair
        root = encoder.mergeEncoder(tl.rootCode, tr.rootCode)
        l, r = decoder.mergeDecoder(root) 
        loss = (torch.norm(l - tl.rootCode) + torch.norm(r - tr.rootCode)).item()
        loss += (tl.loss + tr.loss)
        return (loss, tl, tr, root)

    def makeLeaf (paths, idx, feature) : 
        leaf = nx.DiGraph()
        leaf.add_node(idx, pathSet=[idx])
        tree = Tree(leaf)
        tree.rootCode = feature
        tree.loss = 0
        return tree

    def toLeaves (pathObj) :
        idx, path = pathObj
        desc = [f(path.path, vb) for f in descFunctions] 
        flattened = torch.tensor(list(more_itertools.collapse(desc))).cuda()
        feature = encoder.pathEncoder(flattened)
        leaf = makeLeaf(paths, idx, feature)
        return leaf

    descFunctions = list(map(partial(getattr, Utilities), config['desc_functions']))
    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    vb = doc.get_viewbox()
    totalPaths = len(paths)

    trees = list(map(toLeaves, enumerate(paths)))
    
    candidates = list(map(computeMerges, itertools.combinations(trees, 2)))
    heapq.heapify(candidates)

    while len(candidates) > 0 :
        loss, tl, tr, root = heapq.heappop(candidates)

        if tl not in trees or tr not in trees : 
            continue

        print(tl.nPaths, tr.nPaths)

        trees.remove(tl)
        trees.remove(tr)

        newTree = Tree(mergeTrees([tl.tree, tr.tree]))

        newTree.rootCode = root
        newTree.loss = loss

        trees.append(newTree)

        if newTree.nPaths == totalPaths : 
            return newTree
        else :
            newCombinations = itertools.product(trees[:-1], [newTree])
            newCandidates = map(computeMerges, newCombinations)
            for candidate in newCandidates : 
                heapq.heappush(candidates, candidate)

