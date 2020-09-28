"""
Set of classes for encoding paths. 
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from functools import reduce
import math
from collections.abc import Iterable

class CyclicPathEncoder (nn.Module): 

    def __init__ (self, config) : 
        super(CyclicPathEncoder, self).__init__() 
        self.kernel_size = config['kernel_size']
        self.feature_size = config['feature_size']
        # Perform 1D Convolution using multiple kernels
        self.encoder = nn.Sequential(
            nn.Conv1d(2, self.feature_size, self.kernel_size),
            nn.ReLU()
        )

    def encode (self, pts) : 
        pts = torch.cat((pts, pts[:,:,:self.kernel_size-1]), dim=2)
        encoding = torch.mean(self.encoder(pts), dim=2)
        return encoding

    def forward (self, pts, **kwargs): 
        """
        Expected input shape: N x 2 x nSamples. I think.
        """
        N, _, _ = pts.shape
        encoding = self.encode(pts)
        return encoding

class CyclicPathDecoder (nn.Module) :

    def __init__ (self, config) : 
        super(CyclicPathDecoder, self).__init__()
        samples = config['samples']
        feature_size = config['feature_size']
        hidden_size = config['hidden_size']
        angles = torch.linspace(0, 2 * math.pi, samples).reshape((-1, 1))
        self.points = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
        # Decoder is an MLP which takes the feature extracted
        # by the encoder and a point on a circle and predicts the 
        # ground truth point on the path.
        self.decoder = nn.Sequential(
            nn.Linear(feature_size + 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2),
            nn.Sigmoid()
        )

    def forward (self, encoding, **kwargs) : 
        N, *_ = encoding.shape
        newPts = []
        for pt in self.points :
            reshapedPt = pt.reshape((1, -1))
            repeatedPt = torch.cat([reshapedPt for _ in range(N)])    
            decoderIn = torch.cat((encoding, repeatedPt), dim=1)
            newPt = self.decoder(decoderIn)
            newPts.append(newPt)
        return torch.stack(newPts).permute(1, 2, 0)

class MLPPathEncoder(nn.Module):
    """
    Single Layer NN.

    Used so that the inputs to the merge 
    encoder are of a uniform dimension.
    """

    def __init__(self, config):
        """
        Constructor

        Parameters
        ----------
        input_size : int/tuple
            The dimension of the path descriptor 
        feature_size : int
            The dimension of the feature
        """
        super(MLPPathEncoder, self).__init__()
        input_size = config['input_size']
        feature_size = config['feature_size']
        if isinstance(input_size, Iterable) :
            input_size = reduce(lambda a, b : a * b, input_size)
        self.encoder = nn.Linear(input_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, path_input, **kwargs):
        N, *_ = path_input.shape
        path_vector = self.encoder(path_input.view((N, -1)))
        path_vector = self.tanh(path_vector)
        return path_vector

class MLPPathDecoder(nn.Module):
    """
    Complement of the MLPPathEncoder
    """
    def __init__(self, config):
        super(MLPPathDecoder, self).__init__()
        feature_size = config['feature_size']
        output_size = config['output_size']
        self.output_size = output_size
        if isinstance(output_size, Iterable) :
            output_size = reduce(lambda a, b : a * b, output_size)
        self.mlp = nn.Linear(feature_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature, **kwargs):
        x = self.mlp(parent_feature)
        x = self.tanh(x)
        return x.view((-1, *self.output_size))

class GraphNet (nn.Module) :
    """
    Two layers of GIN Convolution.
    """
    def __init__ (self, config):
        super(GraphNet, self).__init__()
        self.input_size = config['input_size']
        hidden_size = config['hidden_size']
        self.output_size = config['output_size']
        if isinstance(self.input_size, Iterable) :
            input_size = reduce(lambda a, b : a * b, self.input_size)
        else : 
            input_size = self.input_size
        if isinstance(self.output_size, Iterable) :
            output_size = reduce(lambda a, b : a * b, self.output_size)
        else : 
            output_size = self.output_size
        nn1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, output_size)
        )
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index) : 
        N, *_ = x.shape
        if isinstance(self.input_size, Iterable) :
            x = x.view((N, -1))
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        if isinstance(self.output_size, Iterable) :
            return x.view((-1, *self.output_size))
        else :
            return x

if __name__ == "__main__" :
    from descriptor import equiDistantSamples
    import svgpathtools as svg
    from relationshipGraph import *
    doc = svg.Document('/Users/amaltaas/BTP/vectorrvnn/PartNetSubset/Train/10007.svg')
    paths = doc.flatten_all_paths()
    vbox = doc.get_viewbox()
    graph = adjGraph(paths)
    x = torch.tensor([equiDistantSamples(p.path, vbox) for p in paths])
    ######################################################################################
    ## Test Cyclic Convolution.
    ######################################################################################
    e = CyclicPathEncoder({'kernel_size': 3, 'feature_size': 10})
    d = CyclicPathDecoder({'feature_size': 10, 'hidden_size': 50, 'samples': 5})
    model = nn.Sequential(e, d)
    print(x.shape, model(x).shape)
    ######################################################################################
    ## Test MLP. 
    ######################################################################################
    e = MLPPathEncoder({'input_size': (2, 5), 'feature_size': 10})
    d = MLPPathDecoder({'feature_size': 10, 'output_size': (2, 5)})
    model = nn.Sequential(e, d)
    print(x.shape, model(x).shape)
    ######################################################################################
    ## Test GraphNet.
    ######################################################################################
    enc = GraphNet({'input_size': (2, 5), 'hidden_size': 10, 'output_size': 20})
    dec = GraphNet({'input_size': 20, 'hidden_size': 10, 'output_size': (2, 5)})
    edge_indices = torch.tensor(np.array(graph.edges).T)
    h = enc(x, edge_indices)
    o = dec(h, edge_indices)
    print(x.shape, o.shape)

