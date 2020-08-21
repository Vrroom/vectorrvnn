import math
import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torch.autograd import Variable
from functools import reduce
from time import time
from torch_geometric.nn.conv import GINConv

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
        self.encoder = GraphNet(config)
        self.decoder = GraphNet(config)
        dim = 32
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, 11)
        )

    def forward (self, x, edge_index) :
        f = self.encoder(x, edge_index)
        classScores = self.classifier(f)
        x_ = self.decoder(f, edge_index)
        return classScores, x_

if __name__ == "__main__" :
    config = {'input_size': 10}
    model = GraphAutoEncoder(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    creLoss = nn.CrossEntropyLoss()
    mseLoss = nn.MSELoss()
    for data in loader : 
        optimizer.zero_grad()
        classScores, x_ = model(data.x, data.edge_index)
        loss = creLoss(classScores, data.y) + mseLoss(x_, data.x)
        loss.backward()
        optimizer.step()

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
    Takes the left and the right feature
    and outputs a combined feature for both.

    Question: What can be done to make this invariant
    to the order of the subtrees?

    Answer: Obfuscate the inputs to the network.
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

        sizes = hidden_size if type(hidden_size) is list else [hidden_size]
        sizes = [feature_size, *sizes]

        self.left = nn.ModuleList([nn.Linear(a, b) for a, b in zip(sizes, sizes[1:])])
        self.right = nn.ModuleList([nn.Linear(a, b, bias=False) for a, b in zip(sizes, sizes[1:])])

        self.second = nn.Linear(sizes[-1], feature_size)
        self.tanh = nn.Tanh()

    def forward(self, left_input, right_input):
        output = reduce(lambda x, y : y(x), self.left, left_input)
        output += reduce(lambda x, y : y(x), self.right, right_input)
        output = self.tanh(output)
        output = self.second(output)
        output = self.tanh(output)
        return output

class MergeDecoder(nn.Module):
    """
    Complement of the MergeEncoder
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
        super(MergeDecoder, self).__init__()
        sizes = hidden_size if type(hidden_size) is list else [hidden_size]
        sizes = [feature_size, *sizes]

        self.first = nn.Linear(feature_size, sizes[-1])

        revSizes = list(reversed(sizes))
        self.left  = nn.ModuleList([nn.Linear(a, b) for a, b in zip(revSizes, revSizes[1:])])
        self.right = nn.ModuleList([nn.Linear(a, b) for a, b in zip(revSizes, revSizes[1:])])

        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        output = self.tanh(self.first(parent_feature))

        left_feature = reduce(lambda x, y : y(x), self.left, output)
        right_feature = reduce(lambda x, y : y(x), self.right, output)

        left_feature = self.tanh(left_feature)
        right_feature = self.tanh(right_feature)

        return left_feature, right_feature

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

class GRASSAutoEncoder(nn.Module):
    """ 
    RvNN Autoencoder for vector-graphics.
    """
    def __init__(self, config):
        super(GRASSAutoEncoder, self).__init__()

        pathCodeSize = config['path_code_size']
        featureSize  = config['feature_size']
        hiddenSize   = config['hidden_size']

        self.graphNet = GraphNet(config)

        self.path_encoder = PathEncoder(pathCodeSize, featureSize)
        self.path_decoder = PathDecoder(featureSize, pathCodeSize)

        self.merge_encoder = MergeEncoder(featureSize, hiddenSize)
        self.merge_decoder = MergeDecoder(featureSize, hiddenSize)

        self.node_classifier = NodeClassifier(featureSize, hiddenSize)
        self.mse_loss = nn.MSELoss()
        self.cre_loss = nn.CrossEntropyLoss()

        self.raster_encoder = RasterEncoder(featureSize)

    def graphEncoder (self, batch) :
        return self.graphNet(batch.x, batch.edge_index)

    def pathDecoder(self, feature):
        return self.path_decoder(feature)

    def rasterEncoder(self, image) : 
        return self.raster_encoder(image)

    def mergeDecoder(self, feature):
        return self.merge_decoder(feature)

    def mseLoss(self, a, b) : 
        return self.mse_loss(a, b)
    
    def nodeClassifier (self, features, nodeClasses) : 
        labelVectors = self.node_classifier(features)
        a = [self.cre_loss(b.unsqueeze(0), gt).unsqueeze(0) for b, gt in zip(labelVectors, nodeClasses)]
        return torch.cat(a, 0)

    def pathLoss(self, path_feature, gt_path_feature):
        a = [self.mse_loss(b, gt).unsqueeze(0) for b, gt in zip(path_feature, gt_path_feature)]
        return torch.cat(a, 0)

    def nodeLoss(self, node_feature, gt_node_feature):
        a = [self.mse_loss(b, gt).unsqueeze(0) for b, gt in zip(node_feature, gt_node_feature)]
        return torch.cat(a, 0)

    def pathEncoder(self, path):
        return self.path_encoder(path)

    def mergeEncoder(self, left, right):
        return self.merge_encoder(left, right)

    def vectorAdder(self, v1, v2):
        return v1.add_(v2)

def lossFold (fold, tree, image, codes) :

    def encodeNode(node):
        neighbors = list(tree.tree.neighbors(node))
        neighbors.sort()
        isLeaf = len(neighbors) == 0
        if isLeaf:
            pathId = tree.tree.nodes[node]['pathSet'][0]
            path = codes[pathId].reshape((1, -1))
            feature = fold.add('pathEncoder', path)
            features[node] = feature
            return feature
        else : 
            lNode, rNode = neighbors
            left = encodeNode(lNode)
            right = encodeNode(rNode)
            feature = fold.add('mergeEncoder', left, right)
            features[node] = feature
            return feature

    def decodeNode(node, feature):
        neighbors = list(tree.tree.neighbors(node))
        neighbors.sort()
        isLeaf = len(neighbors) == 0
        if isLeaf:
            path = tree.path(node)
            reconPath = fold.add('pathDecoder', feature)
            loss1 = fold.add('pathLoss', path, reconPath) 
            loss2 = fold.add('nodeLoss', features[node], feature)
            classLoss = fold.add('nodeClassifier', feature, torch.zeros((1,1), dtype=torch.long))
            return fold.add('vectorAdder', fold.add('vectorAdder', loss1, loss2), classLoss)
        else :
            lNode, rNode = neighbors
            left, right = fold.add('mergeDecoder', feature).split(2)
            leftLoss = decodeNode(lNode, left)
            rightLoss = decodeNode(rNode, right)
            loss = fold.add('nodeLoss', features[node], feature)
            childLoss = fold.add('vectorAdder', leftLoss, rightLoss)
            classLoss = fold.add('nodeClassifier', feature, torch.ones((1, 1), dtype=torch.long))
            return fold.add('vectorAdder', fold.add('vectorAdder', loss, childLoss), classLoss)

    features = dict()
    root = encodeNode(tree.root)
    target = fold.add('rasterEncoder', image)
    targetLoss = fold.add('nodeLoss', target, root)
    loss = decodeNode(tree.root, root)
    return loss, targetLoss

