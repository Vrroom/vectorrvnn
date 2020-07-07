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
    """
    Construct a hierarchy for a new graphic.

    1. Rasterize the vector graphic.
    2. Use the raster encoder to get a root code.
    3. Decode the root code using decoder.
    4. Make sure that the number of leaves is the 
       same as for the ground truth tree.
    5. Match the leaves in the decoded tree
       with the leaves in the ground truth tree.

    Parameters
    ----------
    gt : Tree
        Ground Truth Tree.
    svgFile : str
        Path to the graphic.
    autoencoder : GRASSAutoEncoder
        Trained model.
    cuda : bool
        Whether to use cuda while evaluating.
    """

    def probability (nodeId, targetClass) : 
        """
        Calculate the probability of target class (one 
        of MERGE or PATH) for the feature at nodeId
        in the tree.
        """
        targetTensor = None
        if targetClass == 0 : 
            targetTensor = torch.zeros((1, 1))
        else : 
            targetTensor = torch.ones((1, 1))
        feature = tree.nodes[nodeId]['feature']
        return autoencoder.nodeClassifier(feature, targetTensor)

    def unroll (feature) :
        """
        Check whether the feature vector 
        represents a merge node and if so,
        find its child features and recursively
        split them.
        """
        nonlocal nodeIdx
        cId = nodeIdx
        tree.add_node(cId)
        tree.nodes[cId]['feature'] = feature
        nodeIdx += 1
        ps = autoencoder.node_classifier(feature)
        if prob(cID, 1) > 0.5 : 
            childFeatures  = autoencoder.mergeDecoder(feature)
            ids = [unroll(f) for f in childFeatures]
            for i in ids :
                tree.add_edge(cId, i)
        return cId

    def extend () :
        """
        Choose the leaf node with minimum probability 
        of being a PATH node among the leaves and 
        split it further. 

        This operation increases the number of leaves
        in the tree by 1.
        """
        nonlocal nodeIdx
        leaves = leaves(tree)
        probs = [probability(l, 0) for l in leaves]
        leafId = argmin(probs)
        childFeatures = autoencoder.mergeDecoder(feature)
        for i in range(nodeIdx, nodeIdx + len(childFeatures)) : 
            tree.add_node(i)
            tree.nodes[i]['feature'] = childFeatures[i]
            tree.add_edge((leafId, i))
        nodeIdx += len(childFeatures)
        
    def prune () :
        """
        From the nodes, all whose children are leaves, 
        choose the node with minimum probability of being
        a MERGE node. Delete the children of this node
        and make this into a leaf node.

        This operation decreases the number of leaves by 1
        for binary trees.
        """
        leaves = leaves(tree)
        bothLeaves = lambda p : len(set(tree.neighbors(p)) & set(leaves)) == 2 
        parents = list(filter(bothLeaves, tree.nodes))
        probs = [probability(p, 1) for p in parents]
        children = list(tree.neighbors(parents[argmin(probs)]))
        tree.remove_nodes_from(children)

    def aggregatePathSets (T, r, neighbors) :
        if T.out_degree(r) > 0 : 
            childrenSets = map(lambda x : T.nodes[x]['pathSet'], neighbors)
            T.nodes[r]['pathSet'] = list(more_itertools.flatten(childrenSets))

    # Get root code from SVG using the rasterEncoder.
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    image = SVGtoNumpyImage(svgFile, H=224, W=224)
    image = torch.from_numpy(image).permute(2, 0, 1)
    image = normalizer(image).unsqueeze(0)
    feature = autoencoder.rasterEncoder(image)

    # Make tree from root code.
    tree = nx.DiGraph()
    tree.add_node(0)
    nodeIdx = 0

    unroll(feature)
    
    # Ensure that the number of leaves in the 
    # constructed tree and in the ground truth
    # tree are the same.
    while len(leaves(gt.tree)) > len(leaves(tree)) : 
        extend()
    while len(leaves(gt.tree)) < len(leaves(tree)) :
        prune()

    # Match the original descriptors to the decoded
    # descriptors.
    costTable = dict()

    gtLeaves = leaves(gt.tree)
    ntLeaves = leaves(tree)

    for i, j in itertools.product(gtLeaves, ntLeaves) : 
        pathIdx = gt.tree.nodes[i]['pathSet'][0]
        feature = tree.nodes[j]['feature']
        vec1 = autoencoder.pathDecoder(feature).detach().numpy()
        vec2 = gt.descriptors[pathIdx].numpy()
        dist = np.linalg.norm(vec1 - vec2)
        costTable[(j, pathIdx)] = dist

    matching = optimalBipartiteMatching(costTable)
    for k, v in matching.items() : 
        if isinstance(k, int) : 
            tree.nodes[k]['pathSet'] = [int(v)]

    # Fill in the tree with pathSet information.
    treeApply(tree, 0, aggregatePathSets)
    return Tree(tree)
