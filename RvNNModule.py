import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

class MLPMergeEncoder (nn.Module): 

    def __init__(self, feature_size, hidden_size):
        super(MLPMergeEncoder, self).__init__()
        self.nn1 = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )

    def forward(self, x) : 
        return torch.sum(self.nn1(x), axis=0)

class MLPMergeDecoder (nn.Module): 

    def __init__(self, feature_size, hidden_size, max_children):
        super(MLPMergeDecoder, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(feature_size, max_children * hidden_size),
            nn.ReLU()
        )
        self.mlp2 = nn.Linear(hidden_size, feature_size)

    def forward(self, x) : 
        x = self.mlp1(parent_feature)
        x = self.mlp2(x.view((-1, self.hidden_size)))
        return x

class GraphMergeEncoder(nn.Module):
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
        self.nn1 = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, feature_size)
        )
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, feature_size)
        )
        self.conv2 = GINConv(nn2)

    def forward(self, x) : 
        N, *_ = x.shape
        edge_index = completeGraph(N)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sum(x, axis=0)

class GraphMergeDecoder(nn.Module):
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
        self.hidden_size = hidden_size
        self.mlp1 = nn.Sequential(
            nn.Linear(feature_size, max_children * hidden_size),
            nn.ReLU()
        )
        self.mlp2 = nn.Linear(hidden_size, feature_size)
        nn1 = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, feature_size)
        )
        self.conv2 = GINConv(nn2)

    def forward(self, parent_feature, tree, node, graph):
        x = self.mlp1(parent_feature)
        x = self.mlp2(x.view((-1, self.hidden_size)))
        return x
        # children_features = torch.stack([m(parent_feature) for m in self.childrenMLPs])
        # # x = F.relu(self.conv1(children_features.squeeze(), self.edge_index))
        # # x = self.bn1(x)
        # # x = F.relu(self.conv2(x, self.edge_index))
        # # x = self.bn2(x)
        # # return x
        # return children_features.squeeze()

