import torch
from sklearn import metrics
from dictOps import *
from functools import lru_cache
from itertools import starmap, combinations
from vis import *
from copy import deepcopy
from treeCompare import * 
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import os.path as osp
import svgpathtools as svg
from treeOps import *
from listOps import *
from graphOps import nxGraph2appGraph
from raster import *
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchUtils import * 
from TripletDataset import *
from torchvision import transforms as T
from more_itertools import collapse
from scipy.cluster.hierarchy import linkage
import torchvision.models as models

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def smallConvNet () : 
    alexnet = models.alexnet(pretrained=True)
    return nn.Sequential(
        alexnet.features,
        nn.Flatten(),
        nn.Linear(256, 128)
    )

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = T.Compose([
    T.ToTensor(),
    whiteBackgroundTransform,
    T.Normalize(mean=mean, std=std),
    lambda t : t.cuda(),
    lambda t : t.unsqueeze(0)
])

@lru_cache(maxsize=128)
def getEmbedding (t, pathSet, embeddingFn): 
    pathSet = asTuple(pathSet)
    allPaths = tuple(leaves(t))
    im    = transform(t.pathSetCrop(allPaths))
    crop  = transform(t.pathSetCrop(pathSet)) 
    whole = transform(t.alphaComposite(pathSet)) 
    return embeddingFn(im, crop, whole) 

class TripletNet (nn.Module) :

    def __init__ (self, config) :
        super(TripletNet, self).__init__() 
        self.hidden_size = config['hidden_size']
        self.conv = smallConvNet()
        self.ALPHA = 0.2
        self.nn = nn.Sequential(
            nn.Linear(3 * 128, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128)
        )

    def embedding (self, im, crop, whole) : 
        imEmbed = self.conv(im)
        cropEmbed = self.conv(crop)
        wholeEmbed = self.conv(whole)
        cat = torch.cat((imEmbed, cropEmbed, wholeEmbed), dim=1)
        return self.nn(cat)

    def forward (self, 
            im,
            refCrop, refWhole, 
            plusCrop, plusWhole, 
            minusCrop, minusWhole,
            refPlus, refMinus) : 
        refEmbed = self.embedding(im, refCrop, refWhole)
        plusEmbed = self.embedding(im, plusCrop, plusWhole)
        minusEmbed = self.embedding(im, minusCrop, minusWhole)
        dplus  = torch.sum((plusEmbed  - refEmbed) ** 2, dim=1, keepdims=True)
        dminus = torch.sum((minusEmbed - refEmbed) ** 2, dim=1, keepdims=True)
        mask = dplus > dminus
        hardRatio = mask.sum() / dplus.shape[0]
        dratio = (dminus[mask] / dplus[mask])
        dplus_ = F.relu((dplus - dminus + self.ALPHA)) # * refMinus / refPlus)
        return dict(dplus_=dplus_, dratio=dratio, hardRatio=hardRatio, mask=mask)

    def greedyTree (self, t, subtrees=None) : 

        def distance (ps1, ps2) : 
            seenPathSets.add(asTuple(ps1))
            seenPathSets.add(asTuple(ps2))
            em1 = getEmbedding(t, ps1, self.embedding) 
            em2 = getEmbedding(t, ps2, self.embedding) 
            return torch.linalg.norm(em1 - em2)

        def subtreeEval (candidate) : 
            childPathSets = [tuple(collapse(c)) for c in candidate]
            return max(starmap(distance, combinations(childPathSets, 2)))

        def simplify (a, b) : 
            candidates = []
            candidatePatterns = ['(*a, *b)', '(a, b)', '(*a, b)', '(a, *b)']
            for pattern in candidatePatterns :
                try : 
                    candidates.append(eval(pattern))
                except Exception : 
                    pass
            scores = list(map(subtreeEval, candidates))
            best = candidates[argmin(scores)]
            return best

        if subtrees is None : 
            subtrees = leaves(t)
        seenPathSets = set()
        with torch.no_grad() : 
            doc = t.doc
            paths = doc.flatten_all_paths()
            while len(subtrees) > 1 : 
                treePairs = list(combinations(subtrees, 2))
                pathSets  = [tuple(collapse(s)) for s in subtrees]
                options   = list(combinations(pathSets, 2))
                distances = list(starmap(distance, options))
                left, right = treePairs[argmin(distances)]
                newSubtree = simplify(left, right)
                subtrees.remove(left)
                subtrees.remove(right)
                subtrees.append(newSubtree)

        return treeFromNestedArray(subtrees)


def testCorrect (model, dataset):  
    dataLoader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        sampler=TripletSampler(dataset.svgDatas, 10000),
        pin_memory=True,
        collate_fn=lambda x : aggregateDict(x, torch.stack)
    )
    for batch in dataLoader : 
        break
    im = batch['im'].cuda()
    refCrop = batch['refCrop'].cuda()
    refWhole = batch['refWhole'].cuda()
    plusCrop = batch['plusCrop'].cuda()
    plusWhole = batch['plusWhole'].cuda()
    minusCrop = batch['minusCrop'].cuda()
    minusWhole = batch['minusWhole'].cuda()
    lcaScore = batch['lcaScore'].cuda()
    dplus2 = model(im, refCrop, refWhole, plusCrop, plusWhole, minusCrop, minusWhole, lcaScore)
    loss = dplus2.mean()
    print(loss)

def fillSVG (gt, t) : 
    doc = svg.Document(gt.svgFile)
    paths = doc.flatten_all_paths()
    vb = doc.get_viewbox()
    for n in t.nodes : 
        t.nodes[n].pop('image', None)
        pathSet = t.nodes[n]['pathSet']
        t.nodes[n]['svg'] = getSubsetSvg2(paths, pathSet, vb)
    thing = treeImageFromGraph(t)
    matplotlibFigureSaver(thing, f'{gt.svgFile}')

def getModel(name) : 
    model = TripletNet(dict(hidden_size=100))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'results', name)
    state_dict = torch.load(os.path.join(MODEL_DIR, "training_end.pth"), map_location=torch.device('cpu'))
    model.load_state_dict(state_dict['model'])
    model = model.float()
    model.to("cuda")
    model.eval()
    return model

if __name__ == "__main__" : 
    with open('cv64.pkl', 'rb') as fp : 
        testData = pickle.load(fp)
    testData = [t for t in testData if t.nPaths < 50]
    model = getModel("suggero_pretrained_alexnet")
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    testData = list(map(treeify, testData))
    inferredTrees = [model.greedyTree(t) for t in tqdm(testData)]
    print(avgMetric(testData, inferredTrees, 1, metrics.fowlkes_mallows_score))
    scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    print(np.mean(scores))
    #for gt, t in tqdm(list(zip(testData, inferredTrees))): 
    #    fillSVG(gt, t)
