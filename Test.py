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
from torchvision import transforms as T
import Utilities
from Utilities import *

def findTree (gt, svgFile, autoencoder, cuda) :

    def unfurl (feature) :
        nonlocal nodeIdx
        cId = nodeIdx
        tree.add_node(cId)
        nodeIdx += 1
        ps = autoencoder.node_classifier(feature)
        if ps[0, 0] > ps[0, 1] : 
            pathFeature = autoencoder.pathDecoder(feature)
            pathFeature = pathFeature.detach().numpy()
            tree.nodes[cId]['desc'] = pathFeature
        else :
            l, r  = autoencoder.mergeDecoder(feature)
            lId = unfurl(l)
            rId = unfurl(r)
            tree.add_edge(cId, lId)
            tree.add_edge(cId, rId)
        return cId

    def aggregatePathSets (T, r, neighbors) :
        if T.out_degree(r) > 0 : 
            childrenSets = map(lambda x : T.nodes[x]['pathSet'], neighbors)
            T.nodes[r]['pathSet'] = list(more_itertools.flatten(childrenSets))

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    image = SVGtoNumpyImage(svgFile, H=224, W=224)
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = normalizer(image).unsqueeze(0)
    feature = autoencoder.rasterEncoder(image)

    tree = nx.DiGraph()
    tree.add_node(0)
    nodeIdx = 0


    costTable = dict()
    gtLeaves = leaves(gt.tree)
    ntLeaves = leaves(tree)
    unfurl(feature)

    for i, j in itertools.product(gtLeaves, ntLeaves) : 
        pathIdx = gt.tree.nodes[i]['pathSet'][0]
        vec1 = tree.nodes[j]['desc']
        vec2 = gt.descriptors[pathIdx].numpy()
        dist = np.linalg.norm(vec1 - vec2)
        costTable[(j, pathIdx)] = dist

    matching = optimalBipartiteMatching(costTable)
    for k, v in matching.items() : 
        if isinstance(k, int) : 
            tree.nodes[k]['pathSet'] = [int(v)]

    treeApply(tree, 0, aggregatePathSets)

    return Tree(tree)
