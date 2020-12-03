import torch
import os
import json
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
from raster import SVGSubset2NumpyImage
from torchUtils import imageForResnet
from tqdm import tqdm
from treeCompare import ted

def subMatrix(mat, rIndices, cIndices) :
    rIndices = torch.tensor(rIndices).cuda()
    cIndices = torch.tensor(cIndices).cuda()
    submat1 = torch.index_select(mat, 0, rIndices)
    submat2 = torch.index_select(submat1, 1, cIndices)
    return submat2

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


    def greedyTree2 (self, t) : 
        subtrees = leaves(t)
        n = len(subtrees)
        doc = svg.Document(t.svgFile)
        imageCache = dict()
        o = self.computeLCAMatrix(t)
        aggregate = torch.min
        while len(subtrees) > 1 : 
            best, bestI, bestJ = np.inf, None, None
            for i in range(len(subtrees)) :
                for j in range(i + 1, len(subtrees)) :
                    subtree1 = subtrees[i]
                    subtree2 = subtrees[j]
                    pathList1 = list(collapse([subtree1]))
                    pathList2 = list(collapse([subtree2]))
                    score = aggregate(subMatrix(o, pathList1, pathList2))
                    if score < best : 
                        best = score
                        bestI = i
                        bestJ = j
            newSubtree = (subtrees[bestI], subtrees[bestJ])
            removeIndices(subtrees, [bestI, bestJ])
            subtrees.append(newSubtree)
        return treeFromNestedArray(subtrees)

    def computeLCAMatrix (self, t, doc) : 

        def getImage (pl) : 
            nonlocal imageCache
            if pl not in imageCache : 
                imageCache[pl] = SVGSubset2NumpyImage(doc, [pl], 224, 224)
            return imageCache[pl]

        n = len(leaves(t)) 
        imageCache = dict()
        o = torch.zeros((n, n)).cuda()
        for i, j in product(range(n), range(n)) :
            img1 = getImage(i)
            img2 = getImage(j)
            img1 = imageForResnet(img1).unsqueeze(0).cuda()
            img2 = imageForResnet(img2).unsqueeze(0).cuda()
            o[i, j] = self.forward(img1, img2)[0, 0]
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
            n = 1 if isinstance(a, int) else len(a)
            m = 1 if isinstance(b, int) else len(b)
            if n + m > 5 or (isinstance(a, int) and isinstance(b, int)): 
                return (a, b)
            else :
                c1 = (a, b) 
                if isinstance(a, int) : 
                    c2 = (a, *b)
                elif isinstance(b, int) :
                    c2 = (*a, b)
                else: 
                    c2 = (*a, *b) 
                if lossHelper(c1) > lossHelper(c2) : 
                    return c2 
                else :
                    return c1
    
        def getImage (subtree) :
            nonlocal imageCache
            pathList = tuple(collapse([subtree]))
            if pathList in imageCache: 
                img = imageCache[pathList]
            else : 
                img = SVGSubset2NumpyImage(doc, pathList, 224, 224)
                imageCache[pathList] = img
            return imageForResnet(img).unsqueeze(0).cuda()

        with torch.no_grad() : 
            subtrees = leaves(t)
            doc = svg.Document(t.svgFile)
            imageCache = dict()
            o = self.computeLCAMatrix(t, doc)
            while len(subtrees) > 1 : 
                best, bestI, bestJ = 1, None, None
                for i in range(len(subtrees)) :
                    for j in range(i + 1, len(subtrees)) :
                        img1 = getImage(subtrees[i])
                        img2 = getImage(subtrees[j])
                        score = float(self.forward(img1, img2)[0, 0])
                        if score < best : 
                            best = score
                            bestI = i
                            bestJ = j
                # Try to simplify the best merge candidate
                newSubtree = simplify(subtrees[bestI], subtrees[bestJ])
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
    with open('o.txt', 'a+') as fd : 
        res = f'{np.mean(scores)} {np.std(scores)}'
        fd.write(res + '\n')
