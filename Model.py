import math
from itertools import product
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
from time import time
from torch_geometric.nn.conv import GINConv
from Utilities import leaves, nonLeaves, optimalBipartiteMatching

def completeGraph (N) :
    """
    Return the edges for a complete graph on N nodes.
    """
    edge_index = torch.tensor(list(product(range(N), range(1, N))))
    edge_index = torch.t(edge_index).long()
    return edge_index

class GraphNet (nn.Module) :
    """
    Two layers of GIN Convolution.
    """
    def __init__ (self, config) :
        super(GraphNet, self).__init__()
        self.input_size = config['path_code_size']
        dim = 32
        nn1 = nn.Sequential(
            nn.Linear(self.input_size, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(dim)
        nn2 = nn.Sequential(
            nn.Linear(dim, dim), 
            nn.ReLU(), 
            nn.Linear(dim, self.input_size)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(self.input_size)

    def forward(self, x, edge_index) : 
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        return x

class GraphAutoEncoder (nn.Module) :

    def __init__ (self, config) :
        super(GraphAutoEncoder, self).__init__()
        self.input_size = config['path_code_size']
        self.encoder = GraphNet(config)
        self.decoder = GraphNet(config)
        dim = 32
        self.classifier = nn.Sequential(
            nn.Linear(self.input_size, dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, 11)
        )

    def forward (self, x, edge_index) :
        f = self.encoder(x, edge_index)
        classScores = self.classifier(f)
        x_ = self.decoder(f, edge_index)
        return classScores, x_

class PathEncoder(nn.Module):
    """
    Single Layer NN.

    Used so that the inputs to the merge 
    encoder are of a uniform dimension.
    """

    def __init__(self, input_size, feature_size):
        """
        Constructor

        Parameters
        ----------
        input_size : int
            The dimension of the path descriptor 
        feature_size : int
            The dimension of the feature
        """
        super(PathEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, path_input):
        path_vector = self.encoder(path_input)
        path_vector = self.tanh(path_vector)
        return path_vector

class MergeEncoder(nn.Module):
    """ 
    This is two layers of GINConv on a complete graph
    followed by a readout operation.
    """

    def __init__(self, feature_size, hidden_size):
        """
        Constructor

        The output dimension is the 
        same as the feature dimension.

        Parameters
        ----------
        feature_size : int
        hidden_size : list or int
            If there is only one hidden layer,
            then we have an int. Else, for 
            multiple layers, we have a list.
        """
        super(MergeEncoder, self).__init__()
        nn1 = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, feature_size)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(feature_size)

    def forward(self, x) : 
        N, *_ = x.shape
        edge_index = completeGraph(N)
        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        return torch.sum(x, axis=0)

if __name__ == "__main__" : 
    net = MergeEncoder(5, 5)
    print(net(torch.ones(2, 5)))

class MergeDecoder(nn.Module):
    """
    Complement of the MergeEncoder
    """
    def __init__(self, feature_size, hidden_size, max_children):
        """
        Constructor

        The output dimension is the 
        same as the feature dimension.

        Parameters
        ----------
        feature_size : int
        hidden_size : list or int
            If there is only one hidden layer,
            then we have an int. Else, for 
            multiple layers, we have a list.
        max_children : int 
            Max sub-trees that are permissible
            per node.
        """
        super(MergeDecoder, self).__init__()
        self.childrenMLPs = [
            nn.Sequential(
                nn.Linear(feature_size, feature_size),
                nn.ReLU(),
            )
            for _ in range(max_children)
        ]
        nn1 = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, feature_size)
        )
        self.conv2 = GINConv(nn2)
        self.bn2 = nn.BatchNorm1d(feature_size)
        self.edge_index = completeGraph(max_children)

    def forward(self, parent_feature):
        children_features = torch.stack([m(parent_feature) for m in self.childrenMLPs])
        x = F.relu(self.conv1(children_features.squeeze(), self.edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, self.edge_index))
        x = self.bn2(x)
        return x

class PathDecoder(nn.Module):
    """
    Complement of the PathEncoder
    """
    def __init__(self, feature_size, path_size):
        super(PathDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, path_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        return vector

class NodeClassifier(nn.Module):

    def __init__(self, feature_size, hidden_size):
        super(NodeClassifier, self).__init__()
        self.mlp1 = nn.Linear(feature_size, hidden_size)
        self.tanh = nn.Tanh()
        self.mlp2 = nn.Linear(hidden_size, 2)

    def forward(self, input_feature):
        output = self.mlp1(input_feature)
        output = self.tanh(output)
        output = self.mlp2(output)
        return output

class RasterEncoder (nn.Module) :
    """
    Take the vector-graphic to the same
    latent space as the GRASS Encoder.
    """
    def __init__ (self, featureSize) :
        super(RasterEncoder, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.mlp = nn.Linear(1000, featureSize)

    def forward (self, image) :
        features = self.resnet18(image)
        return self.mlp(features)

class VectorRvNNAutoEncoder (nn.Module) : 

    def __init__ (self, config) :
        super(VectorRvNNAutoEncoder, self).__init__()
        input_size = config['path_code_size']
        hidden_size = config['hidden_size']
        feature_size = config['feature_size']
        max_children = config['max_children']

        self.rasterEncoder = RasterEncoder(feature_size)

        self.pathEncoder = PathEncoder(input_size, feature_size)
        self.mergeEncoder = MergeEncoder(feature_size, hidden_size)

        self.existClassifier = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )
        self.nodeClassifier = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
        )

        self.pathDecoder = PathDecoder(feature_size, input_size)
        self.mergeDecoder = MergeDecoder(feature_size, hidden_size, max_children)

    def forward (self, tree, image) : 

        def encodeNode (node) : 
            if tree.tree.out_degree(node) > 0 : 
                for child in tree.tree.neighbors(node) : 
                    encodeNode(child)
                childFeatures = torch.stack([encodedFeatures[_] for _ in tree.tree.neighbors(node)])
                feature = self.mergeEncoder(childFeatures.squeeze())
                encodedFeatures[node] = feature.unsqueeze(0)
        
        def decodeNode (node) : 
            if tree.tree.out_degree(node) > 0 :
                feature  = decodedFeatures[node]
                childFeatures = self.mergeDecoder(feature)
                neighbors = list(tree.tree.neighbors(node))
                encodedChildFeatures = [encodedFeatures[_] for _ in neighbors]
                costs = [float(mseLoss(c1.squeeze(), c2)) for c1, c2 in product(encodedChildFeatures, childFeatures)]
                allPairs = product(range(len(encodedChildFeatures)), range(len(childFeatures)))
                costTable = dict(zip(allPairs, costs))
                matching = optimalBipartiteMatching(costTable)
                for i in range(len(encodedChildFeatures)) : 
                    decodedFeatures[neighbors[i]] = childFeatures[int(matching[i])].reshape((1, -1))
                notExistingNodes.extend([cf.reshape((1, -1)) for i, cf in enumerate(childFeatures) if i not in matching.values()])
                for child in neighbors:  
                    decodeNode(child)
            
        mseLoss = nn.MSELoss()
        creLoss = nn.CrossEntropyLoss()
        descriptors = torch.stack([tree.path(l) for l in leaves(tree.tree)])
        pathEncodings = [self.pathEncoder(tree.path(l)) for l in leaves(tree.tree)]
        encodedFeatures = dict(zip(leaves(tree.tree), pathEncodings))
        notExistingNodes = []
        encodeNode(tree.root)
        decodedFeatures = {tree.root: encodedFeatures[tree.root]}
        decodeNode(tree.root)
        leafNodes = torch.stack([decodedFeatures[_] for _ in leaves(tree.tree)])
        nonLeafNodes = torch.stack([decodedFeatures[_] for _ in nonLeaves(tree.tree)])
        pathDecodings = self.pathDecoder(leafNodes)
        rasterEncoding = self.rasterEncoder(image)

        exist = torch.cat([*decodedFeatures.values(), *notExistingNodes])
        existTarget = torch.cat((torch.ones(len(decodedFeatures)), torch.zeros(len(notExistingNodes)))).long()

        nodeType = torch.cat([*leafNodes, *nonLeafNodes])
        nodeTypeTarget = torch.cat((torch.ones(len(leafNodes)), torch.zeros(len(nonLeafNodes)))).long()

        descReconLoss = mseLoss(descriptors, pathDecodings)
        subtreeReconLoss = sum([mseLoss(encodedFeatures[n], decodedFeatures[n]) for n in tree.tree.nodes])
        rasterEncoderLoss = mseLoss(rasterEncoding, encodedFeatures[tree.root])
        nodeExistLoss = creLoss(exist, existTarget)
        nodeTypeLoss = creLoss(nodeType, nodeTypeTarget)

        losses = {
            'Descriptor Reconstruction Loss': descReconLoss, 
            'Subtree Reconstruction Loss': subtreeReconLoss,
            'Raster Encoding Loss': rasterEncoderLoss,
            'Node Existance Prediction Loss': nodeExistLoss,
            'Node Type Prediction Loss': nodeTypeLoss
        }

        return losses
        
        
