import math
import torch
from torch import nn
from torch.autograd import Variable
from time import time

#########################################################################################
## Encoder
#########################################################################################

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
            The dimension of the path descriptor (24D)
        feature_size : int
            The dimension of the feature (80D)
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
    Two layer Recursive block.

    Takes the left and the right feature
    and outputs a combined feature for both.

    Question: What can be done to make this invariant
    to the order of the subtrees?
    """

    def __init__(self, feature_size, hidden_size):
        """
        Constructor

        The output dimension is the 
        same as the feature dimension.

        Parameters
        ----------
        feature_size : int
        hidden_size : int
        """
        super(MergeEncoder, self).__init__()
        self.left = nn.Linear(feature_size, hidden_size)
        self.right = nn.Linear(feature_size, hidden_size, bias=False)
        self.second = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, left_input, right_input):
        output = self.left(left_input)
        output += self.right(right_input)
        output = self.tanh(output)
        output = self.second(output)
        output = self.tanh(output)
        return output

class GRASSEncoder(nn.Module):
    """
    The general encoder. Contains
    instances of the MergeEncoder and
    the PathEncoder
    """

    def __init__(self, config):
        """
        Constructor

        Parameters
        ----------
        config : util.Config
            The configuration class for the experiment.
        """
        super(GRASSEncoder, self).__init__()
        self.path_encoder = PathEncoder(config.path_code_size, config.feature_size)
        self.merge_encoder = MergeEncoder(config.feature_size, config.hidden_size)

    def pathEncoder(self, path):
        return self.path_encoder(path)

    def mergeEncoder(self, left, right):
        return self.merge_encoder(left, right)

def encode_structure_fold(fold, tree):
    """
    Something you have to do 
    to specify the order of operations
    to tensorFold
    """
    def encode_node(node):
        if node.is_leaf():
            return fold.add('pathEncoder', node.path)
        elif node.is_merge():
            left = encode_node(node.left)
            right = encode_node(node.right)
            return fold.add('mergeEncoder', left, right)

    return encode_node(tree.root)

def encode_decode (tree, encoder, decoder) : 
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
        if node.is_leaf() : 
            stack.append(node.path)
            feature = encoder.pathEncoder(node.path)
            return feature
        elif node.is_merge() :
            left = encode_node(node.left)
            right = encode_node(node.right)
            feature = encoder.mergeEncoder(left, right) 
            stack.append(feature)
            return feature

    def decode_node (node, feature) :
        if node.is_leaf() : 
            path = stack.pop()
            feature = decoder.pathDecoder(feature)
            return decoder.pathLossEstimator(path, feature)
        elif node.is_merge() :
            top = stack.pop()
            left, right = decoder.mergeDecoder(feature)
            loss = decode_node(node.right, right)
            loss += decode_node(node.left, left)
            loss += decoder.mseLoss(top, feature)
            return loss

    rootCode = encode_node(tree.root) 
    totalLoss = decode_node(tree.root, rootCode)
    return totalLoss



#########################################################################################
## Decoder
#########################################################################################

class MergeDecoder(nn.Module):
    """
    Complement of the MergeEncoder
    """
    def __init__(self, feature_size, hidden_size):
        super(MergeDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, hidden_size)
        self.mlp_left = nn.Linear(hidden_size, feature_size)
        self.mlp_right = nn.Linear(hidden_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        vector = self.mlp(parent_feature)
        vector = self.tanh(vector)
        left_feature = self.mlp_left(vector)
        left_feature = self.tanh(left_feature)
        right_feature = self.mlp_right(vector)
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

class GRASSDecoder(nn.Module):
    """ 
    Stripped down version of 
    the original GrassDecoder.

    Removed the node classifier 
    because it is quite useless for us.
    """
    def __init__(self, config):
        super(GRASSDecoder, self).__init__()
        self.path_decoder = PathDecoder(config.feature_size, config.path_code_size)
        self.merge_decoder = MergeDecoder(config.feature_size, config.hidden_size)
        self.mseLoss = nn.MSELoss()  # pytorch's mean squared error loss
        self.creLoss = nn.CrossEntropyLoss()  # pytorch's cross entropy loss (NOTE: no softmax is needed before)

    def pathDecoder(self, feature):
        return self.path_decoder(feature)

    def mergeDecoder(self, feature):
        return self.merge_decoder(feature)
    
    def pathLossEstimator(self, path_feature, gt_path_feature):
        return torch.cat([self.mseLoss(b, gt).mul(0.4).unsqueeze(0) for b, gt in zip(path_feature, gt_path_feature)], 0)

    def vectorAdder(self, v1, v2):
        return v1.add_(v2)


def decode_structure_fold(fold, feature, tree):
    """ 
    Encoding the operations 
    suitable for tensorFold
    """
    def decode_node_path(node, feature):
        if node.is_leaf():
            path = fold.add('pathDecoder', feature)
            recon_loss = fold.add('pathLossEstimator', path, node.path)
            return recon_loss
        elif node.is_merge():
            left, right = fold.add('mergeDecoder', feature).split(2)
            left_loss = decode_node_path(node.left, left)
            right_loss = decode_node_path(node.right, right)
            loss = fold.add('vectorAdder', left_loss, right_loss)
            return loss

    loss = decode_node_path(tree.root, feature)
    return loss

