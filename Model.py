import numpy as np
import heapq
from torch.distributions import Categorical
import networkx as nx
from treeOps import *
import PathModules 
from PathModules import *
import RvNNModules
from RvNNModules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from RasterEncoderModule import RasterEncoder
import more_itertools
from itertools import product
from functools import partial
from matching import optimalBipartiteMatching, bestAssignmentCost
from listOps import avg, subsets, isDisjoint, argmin
from losses import iou
from treeCompare import ted

def getModel (module, config) : 
    className = config['type']
    return getattr(module, className)(config)

class SkipLayer (nn.Module) :

    def __init__ (self, input_size, hidden_size, dropout=0.5) : 
        super(SkipLayer, self).__init__()
        self.nn1 = nn.Linear(input_size, hidden_size) 
        self.nn2 = nn.Linear(hidden_size, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x) : 
        h = F.relu(self.nn1(x))
        o = self.dropout(self.nn2(h))
        return x + o

def classifier (config) : 
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    return nn.Sequential(
        SkipLayer(input_size, hidden_size),
        nn.ReLU(),
        SkipLayer(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(input_size, 2),
    )

def bboxModel (config) : 
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.SELU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SELU(),
        nn.Linear(hidden_size, 4),
        nn.Hardsigmoid()
    )

class VectorRvNNAutoEncoder (nn.Module) : 
    """
    Architecture inspired by GRASS and StructureNet 
    to auto-encode bounded branching trees.
    """
    def __init__ (self, config) :
        """
        Constructor. 

        Parameters
        ----------
        config : dict
            Contains information regarding which modules
            to use for current experiment.
        """
        super(VectorRvNNAutoEncoder, self).__init__()
        self.pathEncodeModules = nn.ModuleList(
            [getModel(PathModules, cfg) for cfg in config['path']['pathEncoder']]
        )
        self.pathDecodeModules = nn.ModuleList(
            [getModel(PathModules, cfg) for cfg in config['path']['pathDecoder']]
        )
        self.mergeEncoder = getModel(RvNNModules, config['rvnn']['rvnnEncoder'])
        self.mergeDecoder = getModel(RvNNModules, config['rvnn']['rvnnDecoder'])
        self.K = config['bottom_up_K']
        self.splitter = getModel(RvNNModules, config['rvnn']['splitter'])
        self.rasterEncoder = RasterEncoder(config['raster']['feature_size'])
        self.existClassifier = classifier(config['classifier'])
        self.nodeClassifier = classifier(config['classifier'])
        self.bbox = bboxModel(config['bbox'])
        self.config = config
        self.mseLoss = nn.MSELoss()
        self.creLoss = nn.CrossEntropyLoss()

    def pathEncodingForward (self, x, edge_index) : 
        for module in self.pathEncodeModules :
            x = module(x, edge_index=edge_index)
        return x

    def pathDecodingForward (self, x, edge_index) : 
        for module in self.pathDecodeModules : 
            x = module(x, edge_index=edge_index)
        return x

    def _oneStepReconError (self, childFeatures) : 
        root = self.mergeEncoder(childFeatures)
        childStubs = self.splitter(root) 
        reconstructed = self.mergeDecoder(childStubs)
        return self.matchingCost(childFeatures, reconstructed, self.mseLoss)

    def matchDescriptors (self, a, b, c) : 
        """
        Match two sets of descriptors on the
        basis of mean squared error between them.

        Typically, `a` be an N by F tensor and `b` will
        be an M by F tensor. The matching will match
        the rows of `a` to the rows of `b`, minimizing the
        sum of mean squared error between rows.

        Parameters
        ----------
        a : torch.tensor
        b : torch.tensor
        c : function
            Cost function.
        """
        costs = [float(c(c1.squeeze(), c2.squeeze())) for c1, c2 in product(a, b)]
        allPairs = product(range(len(a)), range(len(b)))
        costTable = dict(zip(allPairs, costs))
        matching = optimalBipartiteMatching(costTable)
        return matching

    def matchingCost (self, a, b, c) : 
        costs = [float(c(c1.squeeze(), c2.squeeze())) for c1, c2 in product(a, b)]
        allPairs = product(range(len(a)), range(len(b)))
        costTable = dict(zip(allPairs, costs))
        return bestAssignmentCost(costTable)

    def exists (self, x) : 
        """
        Use the existClassifier to find out whether the 
        node exists or not.

        Parameters
        ----------
        x : torch.tensor
            The feature vector corresponding to the
            node.
        """
        probs = torch.softmax(self.existClassifier(x), dim=-1)
        return int(Categorical(probs).sample()) == 1

    def isLeaf (self, x) : 
        """
        Use the nodeClassifier to find out whether the 
        node is a leaf or not.

        Parameters
        ----------
        x : torch.tensor
            The feature vector corresponding to the
            node.
        """
        probs = torch.softmax(self.nodeClassifier(x), dim=-1)
        return int(Categorical(probs).sample()) == 1

    def sample (self, tree) : 
        """
        Given a sample document, find the tree 
        decomposition of the document. 

        Parameters
        ----------
        tree : SVGData
            We use only the descriptor information. 
            Don't use the ground truth tree structure 
            while sampling because that is obviously 
            stupid.
        """
        def aggregatePathSets (T, r, neighbors) :
            """
            Aggregate the path indices for each subtree
            root. 

            Each subtree root needs to contain the path
            indices within it so that we can visualize
            the sub-SVG induced by this subtree.
            """
            if T.out_degree(r) > 0 : 
                childrenSets = map(lambda x : T.nodes[x]['pathSet'], neighbors)
                T.nodes[r]['pathSet'] = list(more_itertools.flatten(childrenSets))

        def buildTree (T, feature, parentId, maxDepth) : 
            """
            Recursively build the tree starting from 
            a root feature. 

            The maxDepth variable ensures that we don't
            end up with stack overflow. Hence, we only 
            allow trees to have a fixed depth.
            """
            candidateStubs = self.splitter(feature)
            childStubs = []
            nodeIds = []
            for stub in candidateStubs:  
                if self.exists(stub) : 
                    nodeId = len(T.nodes)
                    nodeIds.append(nodeId)
                    T.add_edge(parentId, nodeId)
                    childStubs.append(stub)
            if len(childStubs) == 0 : 
                return
            childStubs = torch.stack(childStubs)
            edge_index = self.mergeDecoder.edgeInference(childStubs)
            childFeatures = self.mergeDecoder(childStubs, edge_index=edge_index)
            for cs, cf, nodeId in zip(childStubs, childFeatures, nodeIds) : 
                T.nodes[nodeId]['feature'] = cf
                if not self.isLeaf(cs):  
                    buildTree(T, cf, nodeId, maxDepth - 1)

        leavesOfTree = leaves(tree)
        rootFeature = self.rasterEncoder(tree.image)
        targetLeaves = len(leavesOfTree)
        # Keep sampling trees till a tree has 
        # more than requisite number of leaves.
        while True:  
            T = nx.DiGraph()
            T.add_node(0)
            buildTree(T, rootFeature, 0, 6)
            if len(leaves(T)) >= targetLeaves : 
                break
        leavesOfT = leaves(T)
        originalDescriptors = tree.descriptors
        leafFeatures = torch.stack([T.nodes[l]['feature'] for l in leavesOfT])
        reconstructedDescriptors = self.pathDecodingForward(leafFeatures, edge_index=tree.edge_index())
        # Match the leaves with the original descriptors of the graphic. 
        matching = self.matchDescriptors(originalDescriptors, reconstructedDescriptors, self.mseLoss)
        leavesToBeKept = set()
        # Prune all the un-matched leaves in the sampled tree.
        for i in range(len(originalDescriptors)) : 
            l = leavesOfT[matching[i]]
            leavesToBeKept.add(l)
            T.nodes[l]['pathSet'] = tree.nodes[leavesOfTree[i]]['pathSet']
        # Prune all the nodes which only have un-matched leaves.
        hasValidLeaf = lambda n : bool(descendants(T, n).intersection(leavesToBeKept))
        nodesToBeRemoved = set(n for n in T.nodes if not hasValidLeaf(n))
        T.remove_nodes_from(nodesToBeRemoved)
        # This will cause some internal nodes to have only one 
        # out going node. Simplify the tree by eliminating such nodes
        # as they are meaningless to the hierarchy.
        T = removeOneOutDegreeNodesFromTree(T)
        for r in T.nodes : 
            if 'feature' in T.nodes[r]:
                del T.nodes[r]['feature']
        treeApplyChildrenFirst(T, findRoot(T), aggregatePathSets)
        return T

    def bottomUpScore (self, tree) : 
        tree_ = self.lowReconstructionErrorTree(tree)
        return ted(tree, tree_)

    def score (self, tree) :
        tree_ = self.sample(tree)
        return ted(tree, tree_)

    def lowReconstructionErrorTree (self, tree) : 
        descriptors = tree.descriptors
        pathIndices = list(range(tree.nPaths))
        pathEncodings = self.pathEncodingForward(descriptors, edge_index=tree.edge_index())
        error = dict(product(pathIndices, [0]))
        features = dict(zip(pathIndices, pathEncodings))
        K = self.K
        candidates = list(subsets(pathIndices, K))
        stackFeatures = lambda candidate : torch.stack([features[c] for c in candidate])
        while len(candidates) > 0 : 
            oldErrors = [sum([error[c] for c in candidate]) for candidate in candidates]
            newErrors = [self._oneStepReconError(stackFeatures(_)) for _ in candidates]
            totalErrors = [o + n for o, n in zip(oldErrors, newErrors)]
            minErrorIndex = argmin(totalErrors)
            bestCandidate = candidates[minErrorIndex]
            bases = list(filter(lambda k : k not in bestCandidate, error.keys()))
            bases.append(bestCandidate)
            error[bestCandidate] = totalErrors[minErrorIndex]
            features[bestCandidate] = self.mergeEncoder(stackFeatures(bestCandidate))
            for key in bestCandidate : 
                del error[key]
                del features[key]
            candidates = list(subsets(bases, K))
        T = nx.DiGraph()
        while len(bases) > 0 : 
            parent = bases.pop()
            if isinstance(parent, int) :
                pathSet = [parent]
            else : 
                pathSet = list(more_itertools.collapse(parent))
                for child in parent : 
                    T.add_edge(parent, child)
                bases.extend(parent)
            T.nodes[parent]['pathSet'] = pathSet 
        return T

    def classificationAccuracy (self, trees) : 
        """
        There are two classifiers in this model, 
        the node existance classifier and the node
        type classifier. 

        For both, report the avg classification 
        accuracy over all the trees.
        """
        existAcc, typeAcc = [], []
        for tree in trees : 
            info = self._forward(tree)
            eProbs, eTarget = info[1], info[2]
            tProbs, tTarget = info[3], info[4]
            existAcc.append(avg((torch.argmax(eProbs, axis=1) == eTarget).float()))
            typeAcc.append(avg((torch.argmax(tProbs, axis=1) == tTarget).float()))
        existAvg = avg(existAcc)
        typeAvg = avg(typeAcc)
        return {'existAccurarcy': existAvg, 'typeAccuracy': typeAvg}

    def iouAvg (self, trees) : 
        """
        Compute the average intersection over
        union score for all bounding boxes over 
        all trees.
        """
        ious = []
        for tree in trees : 
            info = self._forward(tree)
            boxAndTargets = info[-1]
            iouAvg = avg(list(map(lambda bt : iou(bt[0], bt[1]), boxAndTargets)))
            ious.append(iouAvg)
        return avg(ious)
            
    def iouConsistency (self, trees) : 
        """
        Check how consistent the bounding
        box obtained by the bbox model for
        a node are consistent with the bounding
        boxes of its children and average.
        """
        iouc = []
        for tree in trees:  
            info = self._forward(tree) 
            boxAndTargets = info[-1]
            boxEstimates = dict([(ps, be) for (be, _, ps) in boxAndTargets])
            for n in tree.nodes : 
                if tree.out_degree(n) > 0 and n != tree.root : 
                    estimate = boxEstimates[tree.nodes[n]['pathSet']]
                    childEstimates = [boxEstimates[tree.nodes[c]['pathSet']] for c in tree.neighbors(n)]
                    xm = min(c[0] for c in childEstimates)
                    ym = min(c[1] for c in childEstimates)
                    xM = max(c[0] + c[2] for c in childEstimates)
                    yM = max(c[1] + c[3] for c in childEstimates)
                    iouc.append(iou(estimate, torch.tensor([xm, ym, xM - xm, yM - ym])))
        return avg(iouc)

    def _forward (self, tree) : 
        """
        Forward Pass through the Auto-encoder.

        Parameters
        ----------
        tree : SVGData
        """
        def mapPathSet (iterable) :
            """
            Convenience function for extracting pathset 
            from a list of nodes.
            """
            return map(lambda n : tree.nodes[n]['pathSet'], iterable)

        def encodeNode (node) : 
            """
            Recursively encode nodes.
            """
            if tree.out_degree(node) > 0 : 
                for child in tree.neighbors(node) : 
                    encodeNode(child)
                childFeatures = [encodedFeatures[_] for _ in mapPathSet(tree.neighbors(node))]
                childFeatures = torch.stack(childFeatures)
                edge_index = tree.edgeIndicesAtLevel(node)
                feature = self.mergeEncoder(childFeatures, edge_index=edge_index)
                encodedFeatures[tree.nodes[node]['pathSet']] = feature
        
        def decodeNode (node) : 
            """
            Recurively decode nodes. 

            For each node, the mergeDecoder will give a 
            fixed number of child features.  But not all
            of them exist in the tree. 
            """
            if tree.out_degree(node) > 0 :
                nonlocal totalEdgeLoss
                feature  = decodedFeatures[tree.nodes[node]['pathSet']]
                # The splitter module gives us a vector
                # for each child of this node. We have 
                # to figure out which are legitimate children
                # by matching the bounding boxes. 
                candidates = self.splitter(feature)
                boxEstimates = [self.bbox(candidate) for candidate in candidates]
                boxOriginals = [tree.nodes[n]['bbox'] for n in tree.neighbors(node)]
                matching = self.matchDescriptors(boxOriginals, boxEstimates, lambda a, b : -iou(a, b))
                # Now based on the matching, we have to decide
                # which stubs correspond to existing nodes, non-existing
                # nodes, leaf nodes and internal nodes.
                for i, candidate in enumerate(candidates):  
                    if i in matching.values() : 
                        existingStubs.append(candidate)
                    else : 
                        nonExistingStubs.append(candidate)
                # Now, we find the childStubs and pass them through
                # the merge decoder.
                childStubs = [] 
                neighborPathSets = list(mapPathSet(tree.neighbors(node)))
                for i, ps in enumerate(neighborPathSets) : 
                    candidate = candidates[matching[i]]
                    bAndt = (boxOriginals[i], boxEstimates[matching[i]], ps)
                    boxAndTargets.append(bAndt)
                    childStubs.append(candidate)
                    # Using the length of the pathSet, we can
                    # determine whether the stub belongs to a 
                    # leaf node or an internal node.
                    if len(ps) == 1 : 
                        leafStubs.append(candidate)
                    else : 
                        nonLeafStubs.append(candidate)
                childStubs = torch.stack(childStubs)
                edge_index = tree.edgeIndicesAtLevel(node)
                childFeatures = self.mergeDecoder(childStubs, edge_index=edge_index)
                edgeLoss = self.mergeDecoder.classifierLoss(childStubs, edge_index)
                totalEdgeLoss += edgeLoss
                # Update the dictionary of decoded features.
                decodedFeatures.update(zip(neighborPathSets, childFeatures))
                for child in tree.neighbors(node):  
                    decodeNode(child)
            
        descriptors = tree.descriptors
        pathEncodings = self.pathEncodingForward(descriptors, edge_index=tree.edge_index())
        encodedFeatures = dict(map(lambda a : ((a[0],), a[1]), enumerate(pathEncodings)))
        # Store the features which are known not to
        # exist and use them for training the existence 
        # predictor.
        boxAndTargets = []
        existingStubs, nonExistingStubs = [], []
        leafStubs, nonLeafStubs = [], []
        encodeNode(tree.root)
        rootPathSet = tree.nodes[tree.root]['pathSet']
        decodedFeatures = {rootPathSet : encodedFeatures[rootPathSet]}
        totalEdgeLoss = 0
        decodeNode(tree.root)
        # Use the leaf node features and apply the pathDecoder to 
        # reconstruct the descriptors with which we started this process
        leafNodes = torch.stack([decodedFeatures[(i,)] for i in range(tree.nPaths)])
        pathDecodings = self.pathDecodingForward(leafNodes, edge_index=tree.edge_index())
        # Now, having completed the forward pass, its time for loss
        # computation.
        # Compute the mean squared loss between the root code
        # and the output of the raster encoder.
        rasterEncoding = self.rasterEncoder(tree.image)
        rasterEncoderLoss = self.mseLoss(rasterEncoding, encodedFeatures[rootPathSet].unsqueeze(0))
        # Use the existingStubs and nonExistingStubs for
        # cross entropy loss for the existClassifier.
        existFeatures = torch.stack([*existingStubs, *nonExistingStubs])
        exist = self.existClassifier(existFeatures)
        existTarget = torch.cat((torch.ones(len(existingStubs)), torch.zeros(len(nonExistingStubs)))).long()
        nodeExistLoss = self.creLoss(exist, existTarget) 
        # Similarly, do the same for nodeClassifier.
        nodeFeatures = torch.stack([*leafStubs, *nonLeafStubs])
        nodeType = self.nodeClassifier(nodeFeatures)
        nodeTypeTarget = torch.cat((torch.ones(len(leafStubs)), torch.zeros(len(nonLeafStubs)))).long()
        nodeTypeLoss = self.creLoss(nodeType, nodeTypeTarget) 
        # Check whether the pathDecoding was accurate.
        # Measured as a mean squared loss.
        descReconLoss = self.mseLoss(descriptors, pathDecodings)
        # Reconstruction loss for the inferred bounding boxes.
        bboxLoss = sum([-iou(b, t) for (b, t, _) in boxAndTargets])
        losses = {
            'edgeLoss' : totalEdgeLoss,
            'descReconLoss': descReconLoss, 
            'bboxLoss': bboxLoss,
            'rasterEncoderLoss': rasterEncoderLoss,
            'nodeExistLoss': nodeExistLoss,
            'nodeTypeLoss': nodeTypeLoss
        }
        # Weight the losses as specified in the config file.
        for k in losses.keys() : 
            losses[k] *= self.config['lossWeights'][k]
        return losses, exist, existTarget, nodeType, nodeTypeTarget, boxAndTargets

    def forward (self, tree): 
        return self._forward(tree)[0]

if __name__ == "__main__" : 
    import json
    from SVGData import SVGData
    with open('./Configs/config1.json') as fd:
        config = json.load(fd)
    data = SVGData('/Users/amaltaas/BTP/vectorrvnn/PartNetSubset/Train/10007.svg', "adjGraph", 5)
    data.toTensor()
    model = VectorRvNNAutoEncoder(config)
    print(model.iouConsistency([data]))

