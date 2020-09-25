import math
from Data import Tree
import more_itertools
from torch.distributions import Categorical
from itertools import product
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from functools import reduce
from time import time
from torch_geometric.nn.conv import GINConv
from Utilities import leaves, nonLeaves, descendants, findRoot
from Utilities import optimalBipartiteMatching, treeApplyChildrenFirst
from Utilities import removeOneOutDegreeNodesFromTree, hierarchicalClusterCompareFM
import networkx as nx

class VectorRvNNAutoEncoder (nn.Module) : 
    """
    Architecture inspired by GRASS and StructureNet 
    to auto-encode bounded branching trees.
    """
    def __init__ (self, config) :
        """
        Constructor. 

        The AutoEncoder has 7 components. 

            1. Raster Encoder: Function maps a rasterized 
               graphic to a point in the embedding space
               of the RvNN encoder.
            2. Path Encoder: An MLP for encoding the
               descriptors.
            3. Path Decoder: An MLP for decoding the 
               descriptors.
            4. Merge Encoder: RvNN Encoder. The encoder 
               is a GNN where the vertices are the children
               and there is an edge between each pair of
               vertices.
            5. Merge Decoder: RvNN Decoder. The decoder 
               consists of fixed number of MLPs which 
               decode child features from a single parent
               feature. Then, these child features are passed
               through a GNN to obtain the final features.
            6. Node Existence Classifier: An MLP for predicting
               whether a node actually exists in the tree.
            7. Node Type Classifier: An MLP which predicts whether 
               a node is a leaf node or an internal node.

        Parameters
        ----------
        config : dict
            Contains dimensions information.
        """
        super(VectorRvNNAutoEncoder, self).__init__()
        input_size = config['path_code_size']
        hidden_size = config['hidden_size']
        classifier_hidden_size = config['classifier_hidden_size']
        feature_size = config['feature_size']
        # The upper bound on the arity of the nodes.
        max_children = config['max_children'] 
        # Weight for the classifier MLP loss
        self.predict_loss_weight = config['predict_loss_weight']

        # self.rasterEncoder = RasterEncoder(feature_size)

        self.pathEncoder = PathEncoder(input_size, feature_size)
        self.mergeEncoder = MergeEncoder(feature_size, hidden_size)

        self.existClassifier = nn.Sequential(
            nn.Linear(feature_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, 2)
        )
        self.nodeClassifier = nn.Sequential(
            nn.Linear(feature_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Linear(classifier_hidden_size, 2)
        )

        self.pathDecoder = PathDecoder(feature_size, input_size)
        self.mergeDecoder = MergeDecoder(feature_size, hidden_size, max_children)

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

    def sample (self, tree, image) : 
        """
        Given a sample document, find the tree 
        decomposition of the document. 

        Parameters
        ----------
        tree : Data.Tree
            We use only the descriptor information. 
            Don't use the ground truth tree structure 
            while sampling because that is obviously 
            stupid.
        image : torch.tensor
            Rasterized Vector Graphic.
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

        leavesOfTree = leaves(tree.tree)

        rootFeature = self.rasterEncoder(image).squeeze()
        targetLeaves = len(leavesOfTree)

        # Keep sampling trees till a tree has 
        # more than requisite number of leaves.
        while True:  
            T = nx.DiGraph()
            buildTree(T, rootFeature, -1, 6)
            if len(leaves(T)) >= targetLeaves : 
                break

        leavesOfT = leaves(T)

        originalDescriptors = [tree.path(l) for l in leavesOfTree]
        reconstructedDescriptors = self.pathDecoder(torch.stack([T.nodes[l]['feature'] for l in leavesOfT]))

        # Match the leaves with the original descriptors of the graphic. 
        matching = self.matchDescriptors(originalDescriptors, reconstructedDescriptors)
        leavesToBeKept = set()
        
        # Prune all the un-matched leaves in the sampled tree.
        for i in range(len(originalDescriptors)) : 
            l = leavesOfT[matching[i]]
            leavesToBeKept.add(l)
            T.nodes[l]['pathSet'] = tree.tree.nodes[leavesOfTree[i]]['pathSet']

        # Prune all the nodes which only have un-matched leaves.
        hasValidLeaf = lambda n : bool(descendants(T, n).intersection(leavesToBeKept))
        nodesToBeRemoved = set(n for n in T.nodes if not hasValidLeaf(n))
        T.remove_nodes_from(nodesToBeRemoved)
        # This will cause some internal nodes to have only one 
        # out going node. Simplify the tree by eliminating such nodes
        # as they are meaningless to the hierarchy.
        T = removeOneOutDegreeNodesFromTree(T)

        def delFeatures (T, r, neighbors) : 
            if 'feature' in T.nodes[r]:
                del T.nodes[r]['feature']

        treeApplyChildrenFirst(T, findRoot(T), aggregatePathSets)
        treeApplyChildrenFirst(T, findRoot(T), delFeatures)
        return Tree(T)

    def score (self, tree, image) :
        tree_ = self.sample(tree, image)
        bk = [hierarchicalClusterCompareFM(tree, tree_) for _ in range(4)]
        bk = sum(bk) / 4
        return (bk > 0.7).sum()

    def forward (self, tree, image) : 
        """
        Forward Pass through the Auto-encoder.

        Parameters
        ----------
        tree : Data.Tree
            Ground truth hierarchy.
        image : torch.tensor
            Rasterized Vector Graphic.
        """
        def encodeNode (node) : 
            """
            Recursively encode nodes.
            """
            if tree.tree.out_degree(node) > 0 : 
                for child in tree.tree.neighbors(node) : 
                    encodeNode(child)
                childFeatures = torch.stack([encodedFeatures[_] for _ in tree.tree.neighbors(node)])
                feature = self.mergeEncoder(childFeatures.squeeze())
                encodedFeatures[node] = feature.unsqueeze(0)
        
        def decodeNode (node) : 
            """
            Recurively decode nodes. 

            For each node, the mergeDecoder will give a 
            fixed number of child features.  But not all
            of them exist in the tree. So we find a correspondence
            between the child features as given by the mergeDecoder
            and child features in the tree to determine which
            ones are legit.
            """
            if tree.tree.out_degree(node) > 0 :
                feature  = decodedFeatures[node]
                childFeatures = self.mergeDecoder(feature)
                neighbors = list(tree.tree.neighbors(node))
                encodedChildFeatures = [encodedFeatures[_] for _ in neighbors]
                matching = self.matchDescriptors(encodedChildFeatures, childFeatures)
                for i in range(len(encodedChildFeatures)) : 
                    decodedFeatures[neighbors[i]] = childFeatures[matching[i]].reshape((1, -1))
                for i, cf in enumerate(childFeatures) : 
                    if i not in matching.values() : 
                        notExistingNodes.append(cf.reshape((1, -1)))
                        withSibs.append((cf.reshape((1, -1)), neighbors))
                for child in neighbors:  
                    decodeNode(child)
            
        descriptors = torch.stack([tree.path(l) for l in leaves(tree.tree)])
        pathEncodings = [self.pathEncoder(tree.path(l)) for l in leaves(tree.tree)]

        # Store all the subtree features in a dictionary so that
        # later we can compare with the decoded features and
        # make sure that they match
        encodedFeatures = dict(zip(leaves(tree.tree), pathEncodings))

        # Store the features which are known not to
        # exist and use them for training the existence 
        # predictor.
        notExistingNodes = []
        withSibs = []
        # Perform encoding followed by decoding.
        encodeNode(tree.root)
        decodedFeatures = {tree.root: encodedFeatures[tree.root]}
        decodeNode(tree.root)

        # Separate the leaf and the non-leaf decoded nodes
        # and use them for training node type predictor.
        leafNodes = torch.stack([decodedFeatures[_] for _ in leaves(tree.tree)])
        nonLeafNodes = torch.stack([decodedFeatures[_] for _ in nonLeaves(tree.tree)])

        pathDecodings = self.pathDecoder(leafNodes)
        # rasterEncoding = self.rasterEncoder(image)

        features = torch.cat([*decodedFeatures.values(), *notExistingNodes])
        exist = self.existClassifier(torch.cat([*decodedFeatures.values(), *notExistingNodes]))
        existTarget = torch.cat((torch.ones(len(decodedFeatures)), torch.zeros(len(notExistingNodes)))).long()
         
        nodeType = self.nodeClassifier(torch.cat([*leafNodes, *nonLeafNodes]))
        nodeTypeTarget = torch.cat((torch.ones(len(leafNodes)), torch.zeros(len(nonLeafNodes)))).long()

        # Compute various losses and store in dictionary
        # TODO: May want to use chamfer loss over here.
        descReconLoss = self.mseLoss(descriptors, pathDecodings)
        subtreeReconLoss = sum([self.mseLoss(encodedFeatures[n], decodedFeatures[n]) for n in tree.tree.nodes])
        # rasterEncoderLoss = self.mseLoss(rasterEncoding, encodedFeatures[tree.root])
        nodeExistLoss = self.predict_loss_weight * self.creLoss(exist, existTarget) 
        nodeTypeLoss = self.predict_loss_weight * self.creLoss(nodeType, nodeTypeTarget) 

        diffLoss = 1 / (sum([self.mseLoss(cf, sum([decodedFeatures[n] for n in neighbors]) / len(neighbors)) for cf, neighbors in withSibs]) / len(withSibs))

        losses = {
            'Descriptor': descReconLoss, 
            'Subtree': subtreeReconLoss,
            # 'Raster': rasterEncoderLoss,
            'Existance': nodeExistLoss,
            'Type': nodeTypeLoss,
            'Diff': diffLoss
        }

        return losses# , features, existTarget
