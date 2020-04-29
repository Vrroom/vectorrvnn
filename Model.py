import math
import torch
from torch import nn
from torch.autograd import Variable
from functools import reduce
from time import time

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

class GRASSAutoEncoder(nn.Module):
    """ 
    RvNN Autoencoder for vector-graphics.
    """
    def __init__(self, config):
        super(GRASSAutoEncoder, self).__init__()

        pathCodeSize = config['path_code_size']
        featureSize  = config['feature_size']
        hiddenSize   = config['hidden_size']

        self.path_encoder = PathEncoder(pathCodeSize, featureSize)
        self.path_decoder = PathDecoder(featureSize, pathCodeSize)

        self.merge_encoder = MergeEncoder(featureSize, hiddenSize)
        self.merge_decoder = MergeDecoder(featureSize, hiddenSize)

        self.mse_loss = nn.MSELoss()

    def pathDecoder(self, feature):
        return self.path_decoder(feature)

    def mergeDecoder(self, feature):
        return self.merge_decoder(feature)

    def mseLoss(self, a, b) : 
#        return self.mse_loss(a, b).reshape((1,1))
        return self.mse_loss(a, b)

    def mseLoss2(self, a, b) : 
#        return self.mse_loss(a, b).reshape((1,1))
        return self.mse_loss(a, b).view((1, 1))
    
    def mseLossEstimator(self, path_feature, gt_path_feature):
        a = [self.mse_loss(b, gt).unsqueeze(0) for b, gt in zip(path_feature, gt_path_feature)]
        return torch.cat(a, 0)

    def mseLossEstimator_(self, path_feature, gt_path_feature):
        a = [self.mse_loss(b, gt).unsqueeze(0) for b, gt in zip(path_feature, gt_path_feature)]
        return torch.cat(a, 0)

    def pathEncoder(self, path):
        return self.path_encoder(path)

    def mergeEncoder(self, left, right):
        return self.merge_encoder(left, right)

    def vectorAdder(self, v1, v2):
        return v1.add_(v2)

def lossFold (fold, tree) :

    def encodeNode(node):
        neighbors = list(tree.tree.neighbors(node))
        neighbors.sort()
        isLeaf = len(neighbors) == 0
        if isLeaf:
            path = tree.path(node)
            feature = fold.add('pathEncoder', path)
            rootCodes[node] = feature
            return feature
        else : 
            lNode, rNode = neighbors
            left = encodeNode(lNode)
            right = encodeNode(rNode)
            feature = fold.add('mergeEncoder', left, right)
            rootCodes[node] = feature
            return feature

    def decodeNode(node, feature):
        neighbors = list(tree.tree.neighbors(node))
        neighbors.sort()
        isLeaf = len(neighbors) == 0
        if isLeaf:
            path = tree.path(node)
            reconPath = fold.add('pathDecoder', feature)
            loss1 = fold.add('mseLossEstimator_', path, reconPath) 
            loss2 = fold.add('mseLossEstimator', rootCodes[node], feature)
            return fold.add('vectorAdder', loss1, loss2)
        else :
            lNode, rNode = neighbors
            left, right = fold.add('mergeDecoder', feature).split(2)
            leftLoss = decodeNode(lNode, left)
            rightLoss = decodeNode(rNode, right)
            loss = fold.add('mseLossEstimator', rootCodes[node], feature)
            childLoss = fold.add('vectorAdder', leftLoss, rightLoss)
            return fold.add('vectorAdder', loss, childLoss)

    rootCodes = dict()
    loss = decodeNode(tree.root, encodeNode(tree.root))
    return loss

def treeLoss (tree, autoencoder) : 
    """
    Compute reconstruction loss at 
    each level of the tree. 
    For that we have to store all
    intermediate computations during 
    the encoding stage and pop it while
    decoding to measure the loss

    Parameters
    ----------
    tree : grassdata.Tree
        The RvNN tree
    encoder : GRASSEncoder
    decoder : GRASSDecoder
    """
    stack = []

    def encode_node (node) :
        neighbors = list(tree.tree.neighbors(node))
        neighbors.sort()
        isLeaf = len(neighbors) == 0
        if isLeaf : 
            path = tree.path(node)
            feature = autoencoder.pathEncoder(path)
            stack.append(path)
            stack.append(feature)
            return feature
        else :
            childFeatures = list(map(encode_node, neighbors))
            feature = autoencoder.mergeEncoder(*childFeatures) 
            stack.append(feature)
            return feature

    def decode_node (node, feature) :
        neighbors = list(tree.tree.neighbors(node))
        neighbors.sort()
        isLeaf = len(neighbors) == 0
        if isLeaf : 
            featureOld = stack.pop()
            pathOld = stack.pop()
            path = autoencoder.pathDecoder(feature)
            loss1 = autoencoder.mseLoss(path, pathOld)
            loss2 = autoencoder.mseLoss(feature, featureOld)
            return loss1 + loss2
        else :
            top = stack.pop()
            children = autoencoder.mergeDecoder(feature)
            losses = map(decode_node, reversed(neighbors), reversed(children))
            loss = reduce(lambda x, y : x + y, losses)
            loss += autoencoder.mseLoss(top, feature)
            return loss

    rootCode = encode_node(tree.root) 
    totalLoss = decode_node(tree.root, rootCode)
    return totalLoss
