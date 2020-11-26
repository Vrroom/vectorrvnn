import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import json
from Dataset import SVGDataSet
from PathVAE import PathVAE
import os
from more_itertools import collapse
from listOps import removeIndices
from treeOps import treeFromNestedArray, computeLCAMatrix, numNodes2Binarize
from treeCompare import ted
from tqdm import tqdm

def subMatrix(mat, rIndices, cIndices) :
    rIndices = torch.tensor(rIndices).cuda()
    cIndices = torch.tensor(cIndices).cuda()
    submat1 = torch.index_select(mat, 0, rIndices)
    submat2 = torch.index_select(submat1, 1, cIndices)
    return submat2

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

    def binaryMergeScore (self, graph, i, j, x) :
        o = self.forward(x)
        pathSeti = graph['nodes'][i]['paths']
        pathSetj = graph['nodes'][j]['paths']
        return subMatrix(o, pathSeti, pathSetj).mean()

    def greedyTree (self, x, aggregate=torch.mean): 
        n, *_ = x.shape
        o = self.forward(x)
        subtrees = list(map(lambda x : x, range(n)))
        while len(subtrees) > 1 : 
            best, bestI, bestJ = -np.inf, None, None
            for i in range(len(subtrees)) :
                for j in range(i + 1, len(subtrees)) :
                    subtree1 = subtrees[i]
                    subtree2 = subtrees[j]
                    pathList1 = list(collapse([subtree1]))
                    pathList2 = list(collapse([subtree2]))
                    score = aggregate(subMatrix(o, pathList1, pathList2))
                    if score > best : 
                        best = score
                        bestI = i
                        bestJ = j
            newSubtree = (subtrees[bestI], subtrees[bestJ])
            removeIndices(subtrees, [bestI, bestJ])
            subtrees.append(newSubtree)
        return treeFromNestedArray(subtrees)

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

def distanceFromGroundTruth (model, t, aggregator) : 
    t_ = model.greedyTree(t.descriptors, aggregator)
    a = ted(t, t_)
    return ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())

def test (model, t) :
    t_ = (model.greedyTree(t.descriptors))
    # loss = nn.L1Loss()
    # desc = t.descriptors
    # o = model(desc)
    # print(o)
    # print(t.lcaMatrix)

if __name__ == "__main__" : 
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load all the data
    testDir = commonConfig['test_directory']
    testData = SVGDataSet(testDir, 'adjGraph', 10, useColor=False)
    delta = [numNodes2Binarize(t) for t in testData]
    dist = [n / (2 * t.number_of_nodes() + n) for t, n in zip(testData, delta)]
    print(np.mean(dist), np.std(dist))
    # testData.toTensor(cuda=True)
    # pathVAE = PathVAE(config)
    # model = LCAMatrixModel(pathVAE, config['sampler'])
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # MERGE_OUTPUT = os.path.join(BASE_DIR, "results", "lca_merge")
    # state_dict = torch.load(os.path.join(MERGE_OUTPUT, 'training_end.pth'))
    # model.load_state_dict(state_dict['model'])
    # model.to("cuda")
    # aggregators = [torch.min, torch.max, torch.mean]
    # things = []
    # for a in aggregators : 
    #     scores = [distanceFromGroundTruth(model, t, a) for t in tqdm(testData)]
    #     things.append((np.mean(scores), np.std(scores)))
    # print(things)
