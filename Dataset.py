import pickle
from torchvision import transforms as T
import multiprocessing as mp
from functools import partial
from torchUtils import imageForResnet
from torch.utils import data
from osTools import listdir
import torch
from SVGData import SVGData
from treeOps import *
from itertools import starmap, product
from tqdm import tqdm
import h5py 

class Saveable () :
    def save (self, savePath) : 
        with open(savePath, 'wb') as fd :
            pickle.dump(self, fd)

def generateData (dataDir, pickleFileName) : 
    dataPts = map(listdir, listdir(dataDir))
    dataPts = list(map(lambda x : list(reversed(x)), dataPts))
    with mp.Pool(maxtasksperchild=30) as p : 
        svgDatas = list(p.starmap(partial(SVGData, graph=None, samples=None), dataPts))
    with open(pickleFileName, 'wb') as fd : 
        pickle.dump(svgDatas, fd)

class SVGDataSet (data.Dataset, Saveable) : 
    """
    Pre-processed trees from clustering algorithm.
    """
    def __init__ (self, pickleFileName, transform=None) : 
        super(SVGDataSet, self).__init__() 
        with open(pickleFileName, 'rb') as fd : 
            self.svgDatas = pickle.load(fd) 
        # For torchvision.models
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = T.Normalize(mean=mean, std=std)
        if transform is not None : 
            self.transform = T.Compose([transform, self.transform])
        self.examples = []
        for i, t in enumerate(self.svgDatas) :
            nodesA = list(t.nodes)
            roots = [r for r in t.nodes if t.in_degree(r) == 0]
            forests = [descendants(t, r) for r in roots]
            for a, b in product(nodesA, nodesA[1:]):
                if any([{a, b}.issubset(f) for f in forests]) : 
                    l = lca(t, a, b)
                    if l != a and l != b : 
                        self.examples.append((i, a, b))
        
    def __getitem__ (self, index) :
        tId, a, b = self.examples[index]
        t = self.svgDatas[tId]
        im    = torch.from_numpy(t.image).float().permute((2, 0, 1))
        im1   = torch.from_numpy(t.nodes[a]['image'] ).float().permute((2, 0, 1))
        im2   = torch.from_numpy(t.nodes[b]['image'] ).float().permute((2, 0, 1))
        bbox1 = torch.tensor(t.nodes[a]['bbox']).float()
        bbox2 = torch.tensor(t.nodes[b]['bbox']).float()
        lca   = torch.tensor([lcaScore(t, a, b)]).float()
        lof   = torch.tensor([lofScore(t, a, b)]).float()
        if self.transform is not None : 
            im = self.transform(im)
            im1 = self.transform(im1)
            im2 = self.transform(im2)
        return dict(im=im, im1=im1, im2=im2, bbox1=bbox1, bbox2=bbox2, lca=lca, lof=lof)

    def __len__ (self) : 
        return len(self.examples)

if __name__ == "__main__" : 
    import json
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    generateData(commonConfig['train_directory'], 'train.pkl')
    generateData(commonConfig['test_directory'], 'test.pkl')
    generateData(commonConfig['cv_directory'], 'cv.pkl')
