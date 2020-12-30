import torch
from copy import deepcopy
import pickle
import os
import json
import numpy as np
from osTools import listdir
from itertools import product
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
from raster import SVGSubset2NumpyImage, getSubsetSvg
from torchUtils import imageForResnet
from tqdm import tqdm
from treeCompare import ted
from vis import treeImageFromGraph, matplotlibFigureSaver

def subMatrix(mat, rIndices, cIndices) :
    rIndices = torch.tensor(rIndices).cuda()
    cIndices = torch.tensor(cIndices).cuda()
    submat1 = torch.index_select(mat, 0, rIndices)
    submat2 = torch.index_select(submat1, 1, cIndices)
    return submat2

def convLayer (in_channel, out_channel, kernel_size, stride) :
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride, bias=False),
        nn.BatchNorm2d(out_channel),
        RecordingReLU()
    )

def smallResnet (out_size) : 
    resnet = models.resnet18(pretrained=True)
    children = list(resnet.children())
    model = nn.Sequential(*children[:6])
    model.add_module("avgpool", nn.AdaptiveAvgPool2d((1, 1)))
    model.add_module("flatten", nn.Flatten())
    model.add_module("fc", nn.Linear(128, out_size, bias=True))
    return model

class RecordingReLU (nn.ReLU) : 

    def __init__(self) : 
        super(RecordingReLU, self).__init__()
        self.activated = 0

    def forward (self, input) :
        x = super(RecordingReLU, self).forward(input)
        self.activated = ((x > 0).sum() / x.numel()).item()
        return x

class RecordingModule (nn.Module) : 

    def __init__ (self, module) : 
        super(RecordingModule, self).__init__()
        self.module = module
        self.outNorm = 0

    def __getitem__ (self, i) : 
        return self.module[i]

    def forward(self, x) : 
        o = self.module(x)
        self.outNorm = torch.linalg.norm(o).item()
        return o

def smallConvNet(output_size) : 
    return nn.Sequential(
        convLayer(3, 8, 5, 1),
        convLayer(8, 8, 5, 1),
        convLayer(8, 16, 5, 1),
        convLayer(16, 16, 5, 2),
        convLayer(16, 16, 3, 2),
        convLayer(16, output_size, 3, 2),
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten()
    )

class RasterLCAModel (nn.Module) :

    def __init__ (self, config) :
        super(RasterLCAModel, self).__init__()
        self.hidden_size = config['hidden_size']
        self.input_size = config['input_size']
        self.alexnet1 = RecordingModule(smallResnet(self.input_size))
        # self.alexnet2 = RecordingModule(smallResnet(self.input_size))
        self.boxEncoder = RecordingModule(nn.Sequential(
            nn.Linear(4, self.hidden_size, bias=False), 
            nn.BatchNorm1d(self.hidden_size),
            RecordingReLU(),
            nn.Linear(self.hidden_size, self.input_size, bias=True),
        ))
        self.nn1 = RecordingModule(nn.Sequential(
            nn.Linear(3 * self.input_size, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            RecordingReLU(), 
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            RecordingReLU(), 
            nn.Linear(self.hidden_size, self.input_size, bias=False),
            nn.BatchNorm1d(self.input_size),
            RecordingReLU()
        ))
        self.nn2 = RecordingModule(nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            RecordingReLU(), 
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            RecordingReLU(), 
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid()
        ))
        self.boxPredictor = RecordingModule(nn.Sequential(
            nn.Linear(2 * self.input_size, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            RecordingReLU(), 
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.BatchNorm1d(self.hidden_size),
            RecordingReLU(), 
            nn.Linear(self.hidden_size, 4),
            nn.Sigmoid()
        ))
        # self.normalize = RecordingModule(nn.Sequential(
        #     nn.BatchNorm1d(3 * self.input_size), 
        #     RecordingReLU()
        # ))

    def forward (self, im, im1, im2, box1, box2) : 
        N, *_ = im.shape
        b1 = self.boxEncoder(box1)
        b2 = self.boxEncoder(box2)
        imEncoding = self.alexnet1(im)
        im1Encoding = self.alexnet1(im1)
        im2Encoding = self.alexnet1(im2)
        h1 = torch.cat((imEncoding, im1Encoding, b1), dim=1)
        h2 = torch.cat((imEncoding, im2Encoding, b2), dim=1) 
        bf1 = torch.cat((imEncoding, im1Encoding), dim=1)
        bf2 = torch.cat((imEncoding, im2Encoding), dim=1)
        box1Pred = self.boxPredictor(bf1)
        box2Pred = self.boxPredictor(bf2)
        f1 = self.nn1(h1)
        f2 = self.nn1(h2)
        o = self.nn2(f1 + f2)
        return dict(score=o, box1Pred=box1Pred, box2Pred=box2Pred)

    def computeLCAMatrix (self, t, doc) : 
        n = len(leaves(t)) 
        o = torch.zeros((n, n)).cuda()
        for i, j in product(range(n), range(n)) :
            bbox1 = torch.tensor([t.nodes[i]['bbox']]).float().cuda()
            bbox2 = torch.tensor([t.nodes[i]['bbox']]).float().cuda()
            im = imageForResnet(t.image, True).unsqueeze(0)
            img1 = imageForResnet(t.nodes[i]['image'], True).unsqueeze(0)
            img2 = imageForResnet(t.nodes[j]['image'], True).unsqueeze(0)
            result = self.forward(im, img1, img2, bbox1, bbox2)
            o[i, j] = result['score'][0, 0]
        return o

    def greedyTree (self, t) : 

        def lossHelper (candidate) : 
            t = treeFromNestedArray([candidate])
            setNodeDepths(t)
            l = sorted(list(collapse(candidate)))
            M = torch.zeros((len(l), len(l))).cuda()
            for (i, ti), (j, tj) in product(enumerate(l), enumerate(l)) : 
                d = max(t.nodes[ti]['depth'], t.nodes[tj]['depth'])
                l_ = lca(t, ti, tj)
                M[i, j] = min(1, (d - t.nodes[l_]['depth']) / 10)
            A = subMatrix(o, l, l) 
            A = A * (1 - torch.eye(len(l), len(l)).cuda())
            return F.l1_loss(A, M)

        def simplify (a, b) : 
            return (a, b)
            # n = 1 if isinstance(a, int) else len(a)
            # m = 1 if isinstance(b, int) else len(b)
            # if n + m > 5 or (isinstance(a, int) and isinstance(b, int)): 
            #     return (a, b)
            # else :
            #     c1 = (a, b) 
            #     if isinstance(a, int) : 
            #         c2 = (a, *b)
            #     elif isinstance(b, int) :
            #         c2 = (*a, b)
            #     else: 
            #         c2 = (*a, *b) 
            #     if lossHelper(c1) > lossHelper(c2) : 
            #         return c2 
            #     else :
            #         return c1
    
        def getImage (subtree) :
            nonlocal imageCache
            pathList = tuple(collapse([subtree]))
            if pathList in imageCache: 
                img = imageCache[pathList]
            else : 
                img = SVGSubset2NumpyImage(doc, pathList, 224, 224)
                imageCache[pathList] = img
            return imageForResnet(img, True).unsqueeze(0)

        def getBBox (subtree) : 
            pathList = tuple(collapse([subtree]))
            boxes = np.array([t.nodes[n]['bbox'] for n in pathList])
            xm, ym = boxes[:,0].min(), boxes[:,1].min()
            xM, yM = (boxes[:,0] + boxes[:,2]).max(), (boxes[:,1] + boxes[:,3]).max()
            return torch.tensor([[xm, ym, xM - xm, yM - ym]]).float().cuda()

        with torch.no_grad() : 
            subtrees = leaves(t)
            doc = svg.Document(t.svgFile)
            imageCache = dict()
            o = self.computeLCAMatrix(t, doc)
            im = imageForResnet(t.image, True).unsqueeze(0)
            while len(subtrees) > 1 : 
                best, bestI, bestJ = 1, None, None
                for i in range(len(subtrees)) :
                    for j in range(i + 1, len(subtrees)) :
                        img1 = getImage(subtrees[i])
                        img2 = getImage(subtrees[j])
                        bbox1 = getBBox(subtrees[i])
                        bbox2 = getBBox(subtrees[j])
                        result = self.forward(im, img1, img2, bbox1, bbox2)
                        score = float(result['score'][0, 0])
                        if score < best : 
                            best = score
                            bestI = i
                            bestJ = j
                # Try to simplify the best merge candidate
                newSubtree = simplify(subtrees[bestI], subtrees[bestJ])
                removeIndices(subtrees, [bestI, bestJ])
                subtrees.append(newSubtree)
        return treeFromNestedArray(subtrees)

def distanceFromGroundTruth (model, t) : 
    t_ = model.greedyTree(t)
    return ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())

def treeify (t) : 
    n = t.number_of_nodes()
    t_ = deepcopy(t)
    edges = list(product([n], [r for r in t.nodes if t.in_degree(r) == 0]))
    t_.add_edges_from(edges)
    t_.nodes[n]['pathSet'] = leaves(t)
    return t_

def fillSVG (gt, t) : 
    doc = svg.Document(gt.svgFile)
    paths = doc.flatten_all_paths()
    vb = doc.get_viewbox()
    for n in t.nodes : 
        pathSet = t.nodes[n]['pathSet']
        t.nodes[n]['svg'] = getSubsetSvg(paths, pathSet, vb)
    thing = treeImageFromGraph(t)
    matplotlibFigureSaver(thing, f'{gt.svgFile}')
    
if __name__ == "__main__" : 
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    # Load all the data
    testData = SVGDataSet('cv.pkl').svgDatas
    model = RasterLCAModel(config['sampler'])
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", "prev_expt_transparent_raster")
    state_dict = torch.load(os.path.join(MERGE_OUTPUT, 'training_end.pth'))
    model.load_state_dict(state_dict['model'])
    model = model.float()
    model.to("cuda")
    model.eval()
    inferredTrees = [model.greedyTree(t) for t in tqdm(testData)]
    # with open('infer_val.pkl', 'rb') as fd : 
    #     inferredTrees = pickle.load(fd)

    testData = list(map(treeify, testData))
    for gt, t in zip(testData, inferredTrees) : 
        fillSVG(gt, t)
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    with open('o.txt', 'a+') as fd : 
        res = f'{np.mean(scores)} {np.std(scores)}'
        fd.write(res + '\n')
