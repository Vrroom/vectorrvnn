import numpy as np
from treeOps import leaves, nonLeaves
import PathModules 
from PathModules import *
import RvNNModules
from RvNNModules import *
import torch
import torch.nn as nn
from RasterEncoderModule import RasterEncoder
import more_itertools
from itertools import product
from matching import optimalBipartiteMatching
from listOps import avg
from losses import iou

def getModel (module, config) : 
    className = config['type']
    return getattr(module, className)(config)

def classifier (config) : 
    feature_size = config['input_size']
    hidden_size = config['hidden_size']
    return nn.Sequential(
        nn.Linear(feature_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 2),
    )

def bboxModel (config) : 
    feature_size = config['input_size']
    hidden_size = config['hidden_size']
    return nn.Sequential(
        nn.Linear(feature_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 4),
        nn.Sigmoid()
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
        self.pathEncoder  = getModel(PathModules, config['path']['pathEncoder'])
        self.pathDecoder  = getModel(PathModules, config['path']['pathDecoder'])
        self.mergeEncoder = getModel(RvNNModules, config['rvnn']['rvnnEncoder'])
        self.mergeDecoder = getModel(RvNNModules, config['rvnn']['rvnnDecoder'])
        self.splitter = getModel(RvNNModules, config['rvnn']['splitter'])
        self.rasterEncoder = RasterEncoder(config['raster']['feature_size'])
        self.existClassifier = classifier(config['classifier'])
        self.nodeClassifier = classifier(config['classifier'])
        self.bbox = bboxModel(config['bbox'])
        self.config = config
        self.mseLoss = nn.MSELoss()
        self.creLoss = nn.CrossEntropyLoss()

    def matchDescriptors (self, a, b) : 
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
        """
        costs = [float(self.mseLoss(c1.squeeze(), c2.squeeze())) for c1, c2 in product(a, b)]
        allPairs = product(range(len(a)), range(len(b)))
        costTable = dict(zip(allPairs, costs))
        matching = optimalBipartiteMatching(costTable)
        return matching

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
            if maxDepth == 0: 
                return 
            if self.exists(feature) : 
                nodeId = len(T.nodes)
                T.add_edge(parentId, nodeId)
                T.nodes[nodeId]['feature'] = feature
                if not self.isLeaf(feature):  
                    for childFeature in self.mergeDecoder(feature) : 
                        buildTree(T, childFeature, nodeId, maxDepth-1)

        leavesOfTree = leaves(tree)

        rootFeature = self.rasterEncoder(tree.image).squeeze()
        targetLeaves = len(leavesOfTree)

        # Keep sampling trees till a tree has 
        # more than requisite number of leaves.
        while True:  
            T = nx.DiGraph()
            buildTree(T, rootFeature, -1, 6)
            if len(leaves(T)) >= targetLeaves : 
                break

        leavesOfT = leaves(T)

        originalDescriptors = tree.descriptors
        reconstructedDescriptors = self.pathDecoder(torch.stack([T.nodes[l]['feature'] for l in leavesOfT]))

        # Match the leaves with the original descriptors of the graphic. 
        matching = self.matchDescriptors(originalDescriptors, reconstructedDescriptors)
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
        return Tree(T)

    def score (self, tree, image) :
        tree_ = self.sample(tree, image)
        bk = [hierarchicalClusterCompareFM(tree, tree_) for _ in range(4)]
        bk = sum(bk) / 4
        return (bk > 0.7).sum()

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
            existAcc.append(avg(torch.argmax(eProbs, axis=1) == eTarget))
            typeAcc.append(avg(torch.argmax(tProbs, axis=1) == tTarget))
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
                feature  = decodedFeatures[tree.nodes[node]['pathSet']]
                candidates = self.splitter(feature)
                boxEstimates = [self.bbox(candidate) for candidate in candidates]
                boxOriginals = [tree.nodes[n]['bbox'] for n in tree.neighbors(node)]
                matching = self.matchDescriptors(boxOriginals, boxEstimates)
                childStubs = [] 
                neighborPathSets = list(mapPathSet(tree.neighbors(node)))
                for i, ps in enumerate(neighborPathSets) : 
                    bAndt = (boxOriginals[i], boxEstimates[matching[i]], ps)
                    boxAndTargets.append(bAndt)
                    childStubs.append(candidates[matching[i]])
                for i in range(len(candidates)) : 
                    if i not in matching.values() : 
                        notExistingNodes.append(candidates[i])
                childStubs = torch.stack(childStubs)
                edge_index = tree.edgeIndicesAtLevel(node)
                childFeatures = self.mergeDecoder(childStubs, edge_index=edge_index)
                decodedFeatures.update(zip(neighborPathSets, childFeatures))
                for child in tree.neighbors(node):  
                    decodeNode(child)
            
        descriptors = tree.descriptors
        edge_index = torch.from_numpy(np.array(tree.graph.edges).T).long()
        pathEncodings = self.pathEncoder(descriptors, edge_index=edge_index)
        encodedFeatures = dict(map(lambda a : ((a[0],), a[1]), enumerate(pathEncodings)))
        # Store the features which are known not to
        # exist and use them for training the existence 
        # predictor.
        boxAndTargets = []
        notExistingNodes = []
        encodeNode(tree.root)
        rootPathSet = tree.nodes[tree.root]['pathSet']
        decodedFeatures = {rootPathSet : encodedFeatures[rootPathSet]}
        decodeNode(tree.root)
        # Separate the leaf and the non-leaf decoded nodes
        # and use them for training node type predictor.
        leafNodes = torch.stack([decodedFeatures[_] for _ in mapPathSet(leaves(tree))])
        nonLeafNodes = torch.stack([decodedFeatures[_] for _ in mapPathSet(nonLeaves(tree))])
        pathDecodings = self.pathDecoder(leafNodes, edge_index=edge_index)
        rasterEncoding = self.rasterEncoder(tree.image)
        existFeatures = torch.stack([*decodedFeatures.values(), *notExistingNodes])
        exist = self.existClassifier(existFeatures)
        existTarget = torch.cat((torch.ones(len(decodedFeatures)), torch.zeros(len(notExistingNodes)))).long()
        nodeFeatures = torch.stack([*leafNodes, *nonLeafNodes])
        nodeType = self.nodeClassifier(nodeFeatures)
        nodeTypeTarget = torch.cat((torch.ones(len(leafNodes)), torch.zeros(len(nonLeafNodes)))).long()
        # Compute various losses and store in dictionary
        descReconLoss = self.mseLoss(descriptors, pathDecodings)
        bboxLoss = sum([self.mseLoss(b, t) for (b, t, _) in boxAndTargets])
        rasterEncoderLoss = self.mseLoss(rasterEncoding, encodedFeatures[rootPathSet].unsqueeze(0))
        nodeExistLoss = self.creLoss(exist, existTarget) 
        nodeTypeLoss = self.creLoss(nodeType, nodeTypeTarget) 
        losses = {
            'descReconLoss': descReconLoss, 
            'bboxLoss': bboxLoss,
            'rasterEncoderLoss': rasterEncoderLoss,
            'nodeExistLoss': nodeExistLoss,
            'nodeTypeLoss': nodeTypeLoss
        }
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

