import torch
import os
import json
from osTools import listdir
from more_itertools import collapse
from SVGData import SVGData
import multiprocessing as mp
from functools import partial
import math
from torch import nn
from Dataset import SVGDataSet
from listOps import removeIndices
import torchvision.models as models
import torch.nn.functional as F
from treeOps import *
import svgpathtools as svg
from raster import SVGSubset2NumpyImage
from torchUtils import imageForResnet
from tqdm import tqdm
from treeCompare import ted

class RasterLCAModel (nn.Module) :

    def __init__ (self, config) :
        super(RasterLCAModel, self).__init__()
        self.hidden_size = config['hidden_size']
        self.input_size = config['input_size']
        self.resnet18 = models.alexnet(pretrained=True)
        self.nn1 = nn.Sequential(
            nn.Linear(1000, self.hidden_size),
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

    def greedyTree (self, t) : 
        subtrees = leaves(t)
        doc = svg.Document(t.svgFile)
        imageCache = dict()
        while len(subtrees) > 1 : 
            best, bestI, bestJ = 1, None, None
            for i in range(len(subtrees)) :
                for j in range(i + 1, len(subtrees)) :
                    subtree1 = subtrees[i]
                    subtree2 = subtrees[j]
                    pathList1 = tuple(collapse([subtree1]))
                    pathList2 = tuple(collapse([subtree2]))
                    if pathList1 in imageCache: 
                        img1 = imageCache[pathList1]
                    else : 
                        img1 = SVGSubset2NumpyImage(doc, pathList1, 224, 224)
                        imageCache[pathList1] = img1
                    if pathList2 in imageCache: 
                        img2 = imageCache[pathList2]
                    else : 
                        img2 = SVGSubset2NumpyImage(doc, pathList2, 224, 224)
                        imageCache[pathList2] = img2
                    img1 = imageForResnet(img1).unsqueeze(0).cuda()
                    img2 = imageForResnet(img2).unsqueeze(0).cuda()
                    score = float(self.forward(img1, img2)[0, 0])
                    if score < best : 
                        best = score
                        bestI = i
                        bestJ = j
            newSubtree = (subtrees[bestI], subtrees[bestJ])
            removeIndices(subtrees, [bestI, bestJ])
            subtrees.append(newSubtree)
        return treeFromNestedArray(subtrees)

    def forward (self, im1, im2) : 
        f1 = self.nn1(self.resnet18(im1))
        f2 = self.nn1(self.resnet18(im2))
        o = self.nn2(f1 + f2)
        return o

def distanceFromGroundTruth (model, t) : 
    t_ = model.greedyTree(t)
    a = ted(t, t_)
    return ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())

if __name__ == "__main__" : 
    def collate_fn (batch) : 
        im1 = torch.stack([b[0] for b in batch])
        im2 = torch.stack([b[1] for b in batch])
        y = torch.stack([b[2] for b in batch])
        return im1, im2, y
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load all the data
    dataDir = commonConfig['test_directory']
    dataPts = map(listdir, listdir(dataDir))
    dataPts = map(lambda l : [_ for _ in l if not _.endswith('png')], dataPts)
    dataPts = list(map(lambda x : list(reversed(x)), dataPts))
    with mp.Pool(maxtasksperchild=30) as p : 
        testData = p.starmap(partial(SVGData, graph=None, samples=None), dataPts)
    testData = list(filter(lambda x : maxOutDegree(x) <= 5, testData))
    model = RasterLCAModel(config['sampler'])
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", "expt_alexnet_const_norm_narrow")
    state_dict = torch.load(os.path.join(MERGE_OUTPUT, 'training_end.pth'))
    model.load_state_dict(state_dict['model'])
    model = model.float()
    model.to("cuda")
    model.eval()
    scores = [distanceFromGroundTruth(model, t) for t in tqdm(testData)]
    print(np.mean(scores), np.std(scores))
