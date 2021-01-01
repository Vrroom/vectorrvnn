import pickle
import random
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

class TripletSampler () : 
    """
    Triplets are sampled as follows: 

        1. A random tree is chosen.
        2. A random node is chosen in the tree. This
           is the reference node.
        3. A random sibling of the node is chosen. 
        4. A random node is chosen whose lca distance
           from the reference node is > 2.
    """
    def __init__ (self, data, seed=0, val=False) :
        self.data = data
        self.rng = random.Random(seed)
        self.seed = seed
        self.val = val
        self.i = 0

    def __iter__ (self): 
        return self

    def sampleTree (self) : 
        n = len(self.data)
        idx = self.rng.randint(0, n - 1)
        while self.data[idx].number_of_nodes() < 3 : 
            idx = self.rng.randint(0, n - 1)
        return idx

    def sampleRef (self, idx) : 
        t = self.data[idx]
        refId = self.rng.sample(list(t.nodes), k=1).pop()
        while t.in_degree[refId] == 0 : 
            refId = self.rng.sample(list(t.nodes), k=1).pop()
        return refId

    def samplePlus (self, tId, refId) : 
        plus = list(siblings(self.data[tId], refId))
        return self.rng.sample(plus, k=1).pop()

    def sampleMinus (self, tId, refId) : 
        t = self.data[tId]
        minus = list(set(t.nodes) - (siblings(t, refId).union({refId})))
        return self.rng.sample(minus, k=1).pop()
    
    def __next__ (self) : 
        if self.i < len(self) : 
            self.i += 1
            t = self.sampleTree()
            ref = self.sampleRef(t)
            plus = self.samplePlus(t, ref)
            minus = self.sampleMinus(t, ref)
            print(t, ref, plus, minus)
            return (t, ref, plus, minus)
        else :
            self.i = 0
            if self.val : 
                self.rng = random.Random(self.seed)
            raise StopIteration

    def __len__ (self) : 
        # Fixed number of samples for each epoch.
        return 10

class SVGTripletDataSet (data.Dataset, Saveable) : 
    """
    Pre-processed trees from clustering algorithm.
    """
    def __init__ (self, pickleFileName, transform=None) : 
        super(SVGTripletDataSet, self).__init__() 
        with open(pickleFileName, 'rb') as fd : 
            self.svgDatas = pickle.load(fd) 
        # For torchvision.models
        # TODO : REPLACE THIS WITH DATA MEAN.
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        self.transform = T.Compose([
            lambda t : torch.from_numpy(t),
            lambda t : t.float(),
            lambda t : t.permute((2, 0, 1)),
            T.Normalize(mean=mean, std=std)
        ])
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
        tId, ref, plus, minus = index
        t = self.svgDatas[tId]
        im      = self.transform(t.image)
        imRef   = self.transform(t.nodes[ref  ]['image'])
        imPlus  = self.transform(t.nodes[plus ]['image'])
        imMinus = self.transform(t.nodes[minus]['image'])
        return dict(im=im, imRef=imRef, imPlus=imPlus, imMinus=imMinus)

if __name__ == "__main__" : 
    # import json
    # with open('commonConfig.json') as fd : 
    #     commonConfig = json.load(fd)
    # generateData(commonConfig['train_directory'], 'train.pkl')
    # generateData(commonConfig['test_directory'], 'test.pkl')
    # generateData(commonConfig['cv_directory'], 'cv.pkl')
    data_ = SVGTripletDataSet('cv.pkl')
    dataloader = data.DataLoader(data_, sampler=TripletSampler(data_.svgDatas, val=True), batch_size=10)
    for e in range(2) :
        for batch in dataloader : 
            print(e)
