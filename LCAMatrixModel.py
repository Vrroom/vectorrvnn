import torch
from torch import nn
import torch.nn.functional as F
import json
from Dataset import SVGDataSet
from PathVAE import PathVAE
import os

class LCAMatrixModel (nn.Module) : 

    def __init__ (self, pathVAE, config): 
        super(LCAMatrixModel, self).__init__()
        self.pathVAE = pathVAE
        self.input_size = config['input_size']
        self.hidden_size = config['hidden_size']
        self.nn1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size), 
            nn.SELU(), 
            nn.Linear(self.hidden_size, self.input_size), 
            nn.SELU()
        )
        self.nn2 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size), 
            nn.SELU(), 
            nn.Linear(self.hidden_size, 1), 
            nn.Hardsigmoid()
        )

    def forward (self, x) :
        x, _  = self.pathVAE.encode(x)
        n, _ = x.shape
        a = x.unsqueeze(0).repeat(n, 1, 1)
        b = x.unsqueeze(1).repeat(1, n, 1)
        x = torch.stack((a, b))
        x = x.permute(1, 2, 0, 3)
        x = x.reshape(n * n, -1)
        x1 = self.nn1(x[:, :self.input_size])
        x2 = self.nn1(x[:, self.input_size:])
        x = x1 + x2
        o = self.nn2(x).view(n, n)
        return o

def test (model, t) :
    loss = nn.L1Loss()
    desc = t.descriptors
    o = model(desc)
    print(o)
    print(t.lcaMatrix)

if __name__ == "__main__" : 
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load all the data
    testDir = commonConfig['test_directory']
    testData = SVGDataSet(testDir, 'adjGraph', 10, useColor=False)
    testData.toTensor(cuda=True)
    pathVAE = PathVAE(config)
    model = LCAMatrixModel(pathVAE, config['sampler'])
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", "lca_merge")
    state_dict = torch.load(os.path.join(MERGE_OUTPUT, 'training_end.pth'))
    model.load_state_dict(state_dict['model'])
    model.to("cuda")
    test(model, testData[2])
