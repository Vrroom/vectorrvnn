import torch
from vis import *
from copy import deepcopy
from treeCompare import * 
from tqdm import tqdm
import os
import svgpathtools as svg
from treeOps import *
from listOps import *
from raster import *
import numpy as np
from torch import nn
import torch.nn.functional as F
from torchUtils import * 
from TripletDataset import TripletSVGDataSet
from torchvision import transforms as T
from more_itertools import collapse
from scipy.cluster.hierarchy import linkage

def smallConvNet () : 
    return nn.Sequential(
        convLayer(3, 64, 5, 1),
        nn.MaxPool2d(2),
        convLayer(64, 128, 3, 1), 
        nn.MaxPool2d(2),
        convLayer(128, 256, 3, 1), 
        nn.MaxPool2d(2),
        nn.Conv2d(256, 128, 2),
        nn.Flatten()
    )

class TripletNet (nn.Module) :

    def __init__ (self, config) :
        super(TripletNet, self).__init__() 
        self.hidden_size = config['hidden_size']
        self.conv = smallConvNet()
        self.nn = nn.Sequential(
            nn.Linear(256, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128)
        )

    def embedding (self, im, ref) : 
        global_embed = self.conv(im)
        ref_embed = self.conv(ref)
        ref_embed = self.nn(torch.cat((global_embed, ref_embed), dim=1))
        return ref_embed

    def forward (self, im, ref, plus, minus) : 
        global_embed = self.conv(im)
        ref_embed = self.conv(ref)
        plus_embed = self.conv(plus)
        minus_embed = self.conv(minus)
        ref_embed = self.nn(torch.cat((global_embed, ref_embed), dim=1))
        plus_embed = self.nn(torch.cat((global_embed, plus_embed), dim=1))
        minus_embed = self.nn(torch.cat((global_embed, minus_embed), dim=1))
        dplus  =  torch.sqrt(torch.sum((plus_embed  - ref_embed) ** 2, dim=1, keepdims=True))
        dminus =  torch.sqrt(torch.sum((minus_embed - ref_embed) ** 2, dim=1, keepdims=True))
        o = F.softmax(torch.cat((dplus, dminus), dim=1), dim=1)
        dplus_ = o[:, 0] ** 2
        return dplus_

    def dendrogram (self, t) : 
        def getImage (subtree) :
            nonlocal imageCache
            pathSet = tuple(collapse([subtree]))
            if pathSet in imageCache: 
                img = imageCache[pathSet]
            else : 
                img = SVGSubset2NumpyImage(doc, pathSet, 32, 32)
                imageCache[pathSet] = img
            return transform(img)

        with torch.no_grad(): 
            subtrees = leaves(t)
            doc = svg.Document(t.svgFile)
            imageCache = dict()
            mean = [0.83548355, 0.8292917 , 0.79279226] 
            std = [0.25613376, 0.25492862, 0.28038055]
            transform = T.Compose([
                lambda t : torch.from_numpy(t),
                lambda t : t.float(),
                lambda t : t.permute((2, 0, 1)),
                T.Normalize(mean=mean, std=std),
                lambda t : t.cuda(),
                lambda t : t.unsqueeze(0)
            ])
            im = transform(t.image)
            images = [getImage(s) for s in subtrees]
            embeddings = torch.stack([self.embedding(im, im_).squeeze() for im_ in images])
            embeddings = embeddings.cpu().numpy()
            linkageMatrix = linkage(embeddings, method='centroid')
            for row in linkageMatrix : 
                i, j = row[:2]
                i, j = int(i), int(j)
                subtrees.append((subtrees[i], subtrees[j]))
            return treeFromNestedArray(subtrees[-1:])

    def greedyTree (self, t) : 

        def getImage (subtree) :
            nonlocal imageCache
            pathSet = tuple(collapse([subtree]))
            if pathSet in imageCache: 
                img = imageCache[pathSet]
            else : 
                img = SVGSubset2NumpyImage(doc, pathSet, 32, 32)
                imageCache[pathSet] = img
            return transform(img)

        with torch.no_grad() : 
            subtrees = leaves(t)
            doc = svg.Document(t.svgFile)
            imageCache = dict()
            mean = [0.83548355, 0.8292917 , 0.79279226] 
            std = [0.25613376, 0.25492862, 0.28038055]
            transform = T.Compose([
                lambda t : torch.from_numpy(t),
                lambda t : t.float(),
                lambda t : t.permute((2, 0, 1)),
                T.Normalize(mean=mean, std=std),
                lambda t : t.cuda(),
                lambda t : t.unsqueeze(0)
            ])
            im = transform(t.image)
            while len(subtrees) > 1 : 
                best, bestI, bestJ = np.inf, None, None
                for i in range(len(subtrees)) :
                    for j in range(i + 1, len(subtrees)) :
                        img1 = getImage(subtrees[i])
                        img2 = getImage(subtrees[j])
                        e1 = self.embedding(im, img1)
                        e2 = self.embedding(im, img2)
                        score = float(torch.sqrt(torch.sum((e1 - e2) ** 2)))
                        print(subtrees[i], subtrees[j], score)
                        if score < best : 
                            best = score
                            bestI = i
                            bestJ = j
                # Try to simplify the best merge candidate
                newSubtree = (subtrees[bestI], subtrees[bestJ])
                removeIndices(subtrees, [bestI, bestJ])
                subtrees.append(newSubtree)
        return treeFromNestedArray(subtrees)

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
        t.nodes[n]['svg'] = getSubsetSvg(paths, pathSet, vb)
    thing = treeImageFromGraph(t)
    matplotlibFigureSaver(thing, f'{gt.svgFile}')

if __name__ == "__main__" : 
    # Load all the data
    testData = TripletSVGDataSet('cv32.pkl').svgDatas
    testData = [t for t in testData if len(leaves(t)) < 10]
    model = TripletNet(dict(hidden_size=200))
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", "triplet")
    state_dict = torch.load(os.path.join(MERGE_OUTPUT, 'training_end.pth'))
    model.load_state_dict(state_dict['model'])
    model = model.float()
    model.to("cuda")
    model.eval()
    inferredTrees = [model.dendrogram(t) for t in tqdm(testData)]
    # with open('infer_val.pkl', 'rb') as fd : 
    #     inferredTrees = pickle.load(fd)
    testData = list(map(treeify, testData))
    # for gt, gt in tqdm(list(zip(testData, testData))): 
    #     fillSVG(gt, gt)
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    print(np.mean(scores))
    # with open('o.txt', 'a+') as fd : 
    #     res = f'dendrogram val triplet - {np.mean(scores)} {np.std(scores)}'
    #     fd.write(res + '\n')
