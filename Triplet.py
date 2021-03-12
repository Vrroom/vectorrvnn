import torch
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

def smallConvNet () : 
    return nn.Sequential(
        convLayer(4, 16, 5, 1),
        nn.MaxPool2d(2),
        convLayer(16, 32, 3, 1),
        nn.MaxPool2d(2), 
        # nn.Conv2d(256, 128, 2),
        nn.Flatten(),
        nn.Linear(1152, 128)
    )

mean = [0.17859975,0.16340605,0.12297418,0.35452954]
std = [0.32942199,0.30115585,0.25773552,0.46831796]
transform = T.Compose([
    lambda t : torch.from_numpy(t),
    lambda t : t.float(),
    lambda t : t.permute((2, 0, 1)),
    # lambda t : F.avg_pool2d (t, 2),
    T.Normalize(mean=mean, std=std),
    lambda t : t.cuda(),
    lambda t : t.unsqueeze(0)
])

@lru_cache
def getEmbedding (t, pathSet, embeddingFn): 
    pathSet = asTuple(pathSet)
    im    = transform(t.pathSetCrop(leaves(t)))
    crop  = transform(t.pathSetCrop(pathSet)) 
    whole = transform(t.alphaComposite(pathSet)) 
    return embeddingFn(im, crop, whole) 

class TripletNet (nn.Module) :

    def __init__ (self, config) :
        super(TripletNet, self).__init__() 
        self.hidden_size = config['hidden_size']
        self.conv = smallConvNet()
        self.ALPHA = 1
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
        dplus  = torch.sqrt(1e-5 + torch.sum((plusEmbed  - refEmbed) ** 2, dim=1, keepdims=True))
        dminus = torch.sqrt(1e-5 + torch.sum((minusEmbed - refEmbed) ** 2, dim=1, keepdims=True))
        dplus_ = F.softmax(torch.cat((dplus, dminus), dim=1), dim=1)[:, 0]
        mask = dplus_ > 0.4
        hardRatio = mask.sum() / dplus.shape[0]
        dplus_ = dplus_[mask]
        dratio = (dminus[mask] / dplus[mask])
        dplus_ = dplus_ * refMinus[mask] / refPlus[mask]
        return dict(dplus_=dplus_, dratio=dratio, hardRatio=hardRatio, mask=mask)

    def greedyTree (self, t) : 

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

        seenPathSets = set()
        with torch.no_grad() : 
            subtrees = leaves(t)
            doc = svg.Document(t.svgFile)
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

def treeify (t) : 
    n = t.number_of_nodes()
    t_ = deepcopy(t)
    roots = [r for r in t.nodes if t.in_degree(r) == 0]
    if len(roots) > 1 : 
        edges = list(product([n], roots))
        t_.add_edges_from(edges)
        t_.nodes[n]['pathSet'] = leaves(t)
    return t_

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
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", name)
    state_dict = torch.load(os.path.join(MERGE_OUTPUT, 'epoch_8.pth'))
    model.load_state_dict(state_dict['model'])
    model = model.float()
    model.to("cuda")
    model.eval()
    return model

if __name__ == "__main__" : 
    testData = TripletSVGDataSet('cv.pkl').svgDatas
    testData = [t for t in testData if t.nPaths < 50]
    model = getModel("tripletSuggeroRetrain")
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    testData = list(map(treeify, testData))
    inferredTrees = [model.greedyTree(t) for t in tqdm(testData)]
    scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    print(np.mean(scores))
    with open('triplet_retrain_infer_val.pkl', 'wb') as fd : 
        pickle.dump(inferredTrees, fd)
    for gt, t in tqdm(list(zip(testData, inferredTrees))): 
        fillSVG(gt, t)
