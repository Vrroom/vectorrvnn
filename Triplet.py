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
        convLayer(4, 64, 5, 1),
        nn.MaxPool2d(2),
        convLayer(64, 128, 3, 1),
        # nn.MaxPool2d(2),
        # convLayer(128, 256, 3, 1),
        # nn.MaxPool2d(2),
        # nn.Conv2d(256, 128, 2),
        nn.AdaptiveAvgPool2d((1, 1)), 
        nn.Flatten()
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
def getEmbedding (im, pathSet, doc, embeddingFn): 
    pathSet = asTuple(pathSet)
    crop  = transform(SVGSubset2NumpyImage (doc, pathSet, 64, 64, True)) 
    whole = transform(SVGSubset2NumpyImage2(doc, pathSet, 64, 64, True)) 
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
        globalEmbed = self.conv(im)
        cropEmbed = self.conv(crop)
        wholeEmbed = self.conv(whole)
        embed = self.nn(torch.cat((globalEmbed, cropEmbed, wholeEmbed), dim=1))
        return embed

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
        return dict(dplus_=dplus_, dratio=dratio, hardRatio=hardRatio)

    def dendrogram (self, t) : 
        with torch.no_grad(): 
            subtrees = leaves(t)
            doc = svg.Document(t.svgFile)
            im = transform(t.image)
            embeddings = torch.stack([getEmbedding(im, tuple(collapse(s)), doc, self.embedding).squeeze() for s in subtrees])
            embeddings = embeddings.cpu().numpy()
            linkageMatrix = linkage(embeddings, method='centroid')
            for row in linkageMatrix : 
                i, j = row[:2]
                i, j = int(i), int(j)
                subtrees.append((subtrees[i], subtrees[j]))
            return (treeFromNestedArray(subtrees[-1:]))

    def greedyBinaryTree (self, t) : 

        def distance (ps1, ps2) : 
            em1 = getEmbedding(im, ps1, doc, self.embedding) 
            em2 = getEmbedding(im, ps2, doc, self.embedding) 
            return torch.linalg.norm(em1 - em2)

        def subtreeEval (candidate) : 
            childPathSets = [tuple(collapse(c)) for c in candidate]
            return max(starmap(distance, combinations(childPathSets, 2)))

        def simplify (a, b) : 
            return (a, b)

        with torch.no_grad() : 
            subtrees = leaves(t)
            doc = svg.Document(t.svgFile)
            im = transform(t.image)
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

    def greedyTree (self, t) : 
        # TODO : What if we don't commit to one tree and keep growing multiple trees 
        # parallely.

        def distance (ps1, ps2) : 
            seenPathSets.add(asTuple(ps1))
            seenPathSets.add(asTuple(ps2))
            em1 = getEmbedding(im, ps1, doc, self.embedding) 
            em2 = getEmbedding(im, ps2, doc, self.embedding) 
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
            im = transform(t.image)
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

        allEmbeddings = []
        ims = []
        for ps in seenPathSets : 
            em = getEmbedding(im, ps, doc, self.embedding)
            ims.append(svgStringToBitmap(getSubsetSvg2(paths, ps, doc.get_viewbox()), 32, 32, True))
            allEmbeddings.append(em.cpu().numpy())
        m = TSNE(n_components=2, perplexity=2)
        x = m.fit_transform(np.concatenate(allEmbeddings, axis=0))
        putOnCanvas(x, ims, t.svgFile + '_TSNE.png')
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
    state_dict = torch.load(os.path.join(MERGE_OUTPUT, 'training_end.pth'))
    model.load_state_dict(state_dict['model'])
    model = model.float()
    model.to("cuda")
    model.eval()
    return model

if __name__ == "__main__" : 
    # Load all the data
    # from subprocess import call
    # import json
    # DIR = 'cvForApp'
    testData = TripletSVGDataSet('cv4channel.pkl').svgDatas
    testData = [t for t in testData if t.nPaths < 50]
    model = getModel("rgba")
    # # testCorrect(model, TripletSVGDataSet('cv64.pkl'))
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    testData = list(map(treeify, testData))
    inferredTrees = [model.greedyTree(t) for t in tqdm(testData)]
    scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    print(np.mean(scores))
    # inferredTrees = [model.dendrogram(t) for t in tqdm(testData)]
    # scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    # print(np.mean(scores))
    # inferredTrees = [model.greedyBinaryTree(t) for t in tqdm(testData)]
    # scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    # print(np.mean(scores))
    # idFiles = [osp.join(osp.split(t.svgFile)[0], 'id.txt') for t in testData]
    # for i, (tree, idFile) in enumerate(zip(inferredTrees, idFiles)) :
    #     DATA_DIR = osp.join(DIR, str(i))
    #     os.mkdir(DATA_DIR)
    #     with open(osp.join(DATA_DIR, 'tree.json'), 'w+') as fd: 
    #         json.dump(tree, fd)
    #     call(['cp', idFile, DATA_DIR])

    with open('triplet_rgba_infer_val.pkl', 'wb') as fd : 
        pickle.dump(inferredTrees, fd)
    # with open('triplet_hn_sampling_infer_val.pkl', 'rb') as fd : 
    #     inferredTrees = pickle.load(fd)
    # testData = list(map(treeify, testData))
    for gt, t in tqdm(list(zip(testData, inferredTrees))): 
        fillSVG(gt, t)
    # scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    # scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    # # scores_ = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees_), total=len(testData))]
    # print(np.mean(scores))
    # # with open('o.txt', 'a+') as fd : 
    # #     res = f'dendrogram val triplet - {np.mean(scores)} {np.std(scores)}'
    #     fd.write(res + '\n')
