import torch
from itertools import product
from torch import nn
import torchvision.models as models
import torch.nn.functional as F
from torch_geometric.nn.conv import GINConv
from torch.distributions import Categorical

class Splitter (nn.Module) : 

    def __init__ (self, config) : 
        super(Splitter, self).__init__()
        input_size = config['input_size']
        max_children = config['max_children']
        self.output_size = config['output_size']
        self.nn1 = nn.Sequential(
            nn.Linear(input_size, max_children * self.output_size),
            nn.ReLU(),
        )

    def forward (self, x) : 
        return self.nn1(x).view((-1, self.output_size))

class MLPMergeEncoder (nn.Module): 

    def __init__(self, config):
        super(MLPMergeEncoder, self).__init__()
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        self.nn1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, **kwargs) : 
        return torch.sum(self.nn1(x), axis=0)

class MLPMergeDecoder (nn.Module): 

    def __init__(self, config):
        super(MLPMergeDecoder, self).__init__()
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size)
        )

    def forward(self, x, **kwargs) : 
        return self.mlp(x)

# TODO: StructureNet had skip modules.
class GraphMergeEncoder(nn.Module):
    """ 
    This is two layers of GINConv
    followed by a readout operation.
    """

    def __init__(self, config):
        """
        Constructor

        The output dimension is the 
        same as the input dimension.
        """
        super(GraphMergeEncoder, self).__init__()
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        nn1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, input_size)
        )
        self.conv2 = GINConv(nn2)

    def forward(self, x, edge_index) : 
        N, *_ = x.shape
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return torch.sum(x, axis=0)

class EdgeClassifier (nn.Module) : 
    
    def __init__ (self, config) :
        super(EdgeClassifier, self).__init__()
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        self.nn1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.nn2 = nn.Linear(hidden_size, 2)

    def forward (self, x) :
        x1, x2 = torch.split(x, 1)
        f1 = self.nn1(x1)
        f2 = self.nn1(x2)
        return self.nn2(f1 + f2)

class GraphMergeDecoder (nn.Module) : 

    def __init__ (self, config) :
        super(GraphMergeDecoder, self).__init__() 
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        self.edgeExistClassifier = EdgeClassifier(config)
        nn1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        self.conv1 = GINConv(nn1)
        nn2 = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), 
            nn.ReLU(), 
            nn.Linear(hidden_size, input_size)
        )
        self.conv2 = GINConv(nn2)
        self.creLoss = nn.CrossEntropyLoss()

    def edgeInference (self, x) : 
        x_ = enumerate(x)
        edge_list = []
        for (i, f1), (j, f2) in product(x_, x_[1:]) : 
            ef = torch.cat((f1, f2))
            scores = self.edgeExistClassifier(ef)
            probs = torch.softmax(scores, dim=-1)
            if int(Categorical(probs).sample()) == 1 :
                edge_list.append((i, j))
        edge_index = torch.tensor(edge_list).t()
        return edge_index

    def classifierLoss (self, x, presentEdges) :
        n, *_ = x.shape
        presentEdges = set(presentEdges.t().tolist())
        edgeX, y = [], []
        for i, j in product(range(n), range(1, n)) :
            edgeX.append(torch.stack((x[i], x[j])))
            if (i, j) in presentEdges or (j, i) in presentEdges : 
                y.append(1)       
            else : 
                y.append(0)
        edgeX = torch.cat(edgeX)
        y = torch.cat(y)
        return self.creLoss(edgeX, y)

    def forward(self, x, edge_index) : 
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        return x + x1 + x2

if __name__ == "__main__" : 
    from descriptor import equiDistantSamples
    import svgpathtools as svg
    from relationshipGraph import *
    doc = svg.Document('/Users/amaltaas/BTP/vectorrvnn/PartNetSubset/Train/10007.svg')
    paths = doc.flatten_all_paths()
    vbox = doc.get_viewbox()
    graph = adjGraph(paths)
    x = torch.tensor([equiDistantSamples(p.path, vbox) for p in paths])
    N, *_ = x.shape
    x = x.reshape((N, -1))
    ######################################################################################
    ## Test MLP RvNN
    ######################################################################################
    # TODO: RUN TESTS AGAIN
    e = MLPMergeEncoder(10, 100)
    d = MLPMergeDecoder(10, 100, 7)
    print(x.shape, d(e(x)).shape)
    ######################################################################################
    ## Test Graph RvNN
    ######################################################################################
    e = GraphMergeEncoder(10, 100)
    edge_index = torch.tensor(np.array(graph.edges).T)
    print(x.shape, (e(x, edge_index)).shape)
     
