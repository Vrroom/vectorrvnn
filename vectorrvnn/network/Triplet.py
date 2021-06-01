import torch
from sklearn import metrics
from vectorrvnn.utils import *
from functools import lru_cache
from itertools import starmap, combinations
from copy import deepcopy
from sklearn.manifold import TSNE
from tqdm import tqdm
import os
import os.path as osp
import svgpathtools as svg
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchUtils import * 
from TripletDataset import *
from torchvision.models import resnet50
from more_itertools import collapse
from scipy.cluster.hierarchy import linkage
import torchvision.models as models
from PositionalEncoding import PositionalEncoding
import Constants as C
import albumentations as A
from albumentations.pytorch import ToTensorV2

def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def convnet () : 
    # Load BAM! pretrained resnet50
    BASE_DIR = os.path.dirname(os.path.abspath(''))
    MODEL_DIR = os.path.join(BASE_DIR, 'vectorrvnn', 'results', 'bam_aug2')
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 20)
    state_dict = torch.load(os.path.join(MODEL_DIR, "epoch_15.pth"))
    model.load_state_dict(state_dict['model'])
    # Use the weights of the pretrained model to 
    # create weights for the new model.
    stitchedWts = model.fc.weight.repeat((C.embedding_size // 20 + 1, 1))[:C.embedding_size, :]
    stitchedBias = model.fc.bias.repeat(C.embedding_size // 20 + 1)[:C.embedding_size]
    model.fc = nn.Linear(2048, C.embedding_size)
    model.fc.weight.data = stitchedWts
    model.fc.bias.data = stitchedBias
    # Use the convolutional part of resnet for feature extraction 
    # only. Train only the fully connected layer on top.
    set_parameter_requires_grad(model, False)
    set_parameter_requires_grad(model.layer4, True)
    set_parameter_requires_grad(model.fc, True)
    # Make sure that parameters are floats 
    # And load model on cuda!
    model = model.float()
    model.to("cuda")
    return model

transform = A.Compose([
    A.Normalize(mean=C.mean, std=C.std, max_pixel_value=1),
    ToTensorV2()
])

@lru_cache(maxsize=128)
def getEmbedding (t, pathSet, embeddingFn): 
    pathSet = asTuple(pathSet)
    allPaths = tuple(leaves(t))
    im    = whiteBackgroundTransform(t.alphaComposite(allPaths))
    crop  = whiteBackgroundTransform(t.pathSetCrop(pathSet))
    whole = whiteBackgroundTransform(t.alphaComposite(pathSet))
    
    im    = transform(image=im   )['image'].cuda().unsqueeze(0)
    crop  = transform(image=crop )['image'].cuda().unsqueeze(0)
    whole = transform(image=whole)['image'].cuda().unsqueeze(0)
    paddedPaths = (list(allPaths) + [-1] * (C.max_len - len(allPaths)))
    position = torch.tensor(paddedPaths, dtype=torch.long).cuda().unsqueeze(0)   
    return embeddingFn(im, crop, whole, position) 

class TripletNet (nn.Module) :

    def __init__ (self, config) :
        super(TripletNet, self).__init__() 
        self.hidden_size = config['hidden_size']
        self.conv = convnet()
        self.pe = PositionalEncoding()
        self.nn = nn.Sequential(
            nn.Linear(3 * C.embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, C.embedding_size)
        )

    def embedding (self, im, crop, whole, position) : 
        imEmbed = self.conv(im)
        cropEmbed = self.pe(self.conv(crop), position)
        wholeEmbed = self.pe(self.conv(whole), position)
        cat = torch.cat((imEmbed, cropEmbed, wholeEmbed), dim=1)
        return self.nn(cat)

    def forward (self, 
            im,
            refCrop, refWhole, refPosition,
            plusCrop, plusWhole, plusPosition,
            minusCrop, minusWhole, minusPosition,
            refPlus, refMinus) : 
        # TODO: Put a TopK based loss here!!
        refEmbed = self.embedding(im, refCrop, refWhole, refPosition)
        plusEmbed = self.embedding(im, plusCrop, plusWhole, plusPosition)
        minusEmbed = self.embedding(im, minusCrop, minusWhole, minusPosition)
        dplus  = torch.sum((plusEmbed  - refEmbed) ** 2, dim=1, keepdims=True)
        dminus = torch.sum((minusEmbed - refEmbed) ** 2, dim=1, keepdims=True)
        dplus_ = F.softmax(torch.cat((dplus, dminus), dim=1), dim=1)[:, 0]
        mask = dplus_ > 0.4
        hardRatio = mask.sum() / dplus.shape[0]
        dplus_ = dplus_[mask]
        dratio = (dminus[mask] / dplus[mask])
        dplus_ = dplus_ * refMinus[mask] / refPlus[mask]
        return dict(dplus_=dplus_, dratio=dratio, hardRatio=hardRatio, mask=mask)

    def greedyTree (self, t, subtrees=None, binary=False) : 

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
            if binary : return (a, b)
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
        t.nodes[n]['svg'] = getSubsetSvg2(doc, paths, pathSet, vb)
    thing = treeImageFromGraph(t)
    return thing

def getModel(name, version='training_end.pth') : 
    model = TripletNet(dict(hidden_size=100))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = os.path.join(BASE_DIR, 'results', name)
    state_dict = torch.load(os.path.join(MODEL_DIR, version))
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
