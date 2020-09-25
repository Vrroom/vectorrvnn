"""
Set of classes for encoding paths. 
"""
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv

class CyclicPathEncoder (nn.Module): 

    def __init__ (self, kernel_size, feature_size) : 
        super(CyclicPathEncoder, self).__init__() 
        self.kernel_size = kernel_size
        self.feature_size = feature_size
        # Perform 1D Convolution using multiple kernels
        self.encoder = nn.Sequential(
            nn.Conv1d(2, self.feature_size, self.kernel_size),
            nn.ReLU()
        )

    def encode (self, pts) : 
        pts = torch.cat((pts, pts[:,:,:self.kernel_size-1]), dim=2)
        encoding = torch.mean(self.encoder(pts), dim=2)
        return encoding

    def forward (self, pts): 
        """
        Expected input shape: N x 2 x nSamples. I think.
        """
        N, _, _ = pts.shape
        encoding = self.encode(pts)
        return encoding

class CyclicPathDecoder (nn.Module) :

    def __init__ (self, config) : 
        super(CyclicPathDecoder, self).__init__()
        angles = torch.linspace(0, 2 * math.pi, config['samples']).reshape((-1, 1))
        self.points = torch.cat((torch.cos(angles), torch.sin(angles)), dim=1)
        # Decoder is an MLP which takes the feature extracted
        # by the encoder and a point on a circle and predicts the 
        # ground truth point on the path.
        self.decoder = nn.Sequential(
            nn.Linear(self.feature_size + 2, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 2),
            nn.Sigmoid()
        )

    def forward (self, pts) : 
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
        super(MLPPathEncoder, self).__init__()
        self.encoder = nn.Linear(input_size, feature_size)
        self.tanh = nn.Tanh()

    def forward(self, path_input):
        path_vector = self.encoder(path_input)
        path_vector = self.tanh(path_vector)
        return path_vector

class MLPPathDecoder(nn.Module):
    """
    Complement of the MLPPathEncoder
    """
    def __init__(self, feature_size, output_size):
        super(MLPPathDecoder, self).__init__()
        self.mlp = nn.Linear(feature_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, parent_feature):
        x = self.mlp(parent_feature)
        x = self.tanh(x)
        return x

class GraphNet (nn.Module) :
    """
    Two layers of GIN Convolution.
    """
    def __init__ (self, input_size, hidden_size, output_size) :
        super(GraphNet, self).__init__()
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
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return x
