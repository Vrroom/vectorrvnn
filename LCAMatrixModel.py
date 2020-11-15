import torch
from torch import nn
import torch.nn.functional as F
import json
from Dataset import SVGDataSet
from PathVAE import PathVAE
import os

def subMatrix(mat, rIndices, cIndices) : 
    rIndices = torch.tensor(rIndices)
    cIndices = torch.tensor(cIndices)
    submat1 = torch.index_select(mat, 0, rIndices)
    submat2 = torch.index_select(submat1, 1, cIndices)
    return submat2

class LCAMatrixModel (nn.Module) : 

    def __init__ (self, pathVAE, config): 
        super(LCAMatrixModel, self).__init__()
        self.pathVAE = pathVAE
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        self.nn = nn.Sequential(
            nn.Linear(2 * input_size, hidden_size), 
            nn.SELU(), 
            nn.Linear(hidden_size, 1), 
            nn.Hardsigmoid()
        )

    def binaryMergeScore (self, graph, i, j, x) : 
        o = self.forward(x)
        pathSeti = graph['nodes'][i]['paths']
        pathSetj = graph['nodes'][j]['paths']
        return -subMatrix(o, pathSeti, pathSetj).mean()

    def forward (self, x) :
        x, _  = self.pathVAE.encode(x)
        n, _ = x.shape
        a = x.unsqueeze(0).repeat(n, 1, 1)
        b = x.unsqueeze(1).repeat(1, n, 1)
        x = torch.stack((a, b))
        x = x.permute(1, 2, 0, 3)
        x = x.reshape(n * n, -1)
        o = self.nn(x).view(n, n)
        return o
        
if __name__ == "__main__" : 
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load all the data
    trainDir = commonConfig['train_directory']
    trainData = SVGDataSet(trainDir, 'adjGraph', 10, useColor=False)
    trainData.toTensor()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VAE_OUTPUT = os.path.join(BASE_DIR, "results", "path_vae")
    pathVAE = PathVAE(config)
    model = LCAMatrixModel(pathVAE, config['sampler'])
    o = model(trainData[0].descriptors)


