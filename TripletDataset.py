import pickle
import random
from torchvision import transforms as T
import multiprocessing as mp
from functools import partial
from torchUtils import imageForResnet
from torch.utils import data
import torch.nn.functional as F
from osTools import listdir
import torch
from TripletSVGData import TripletSVGData
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
    removeTxt = lambda x : filter(lambda y : not y.endswith('txt'), x)
    dataPts = list(map(lambda x : list(removeTxt(reversed(x))), dataPts))
    with mp.Pool(maxtasksperchild=30) as p : 
        svgDatas = list(p.starmap(partial(TripletSVGData, graph=None, samples=None), dataPts))
    with open(pickleFileName, 'wb') as fd : 
        pickle.dump(svgDatas, fd)

class TripletSampler () : 
    def __init__ (self, data, length, seed=0, val=False) :
        self.data = data
        self.rng = random.Random(seed)
        self.seed = seed
        self.val = val
        self.i = 0
        self.length = length

    def __iter__ (self): 
        return self

    def getSample (self) : 
        n = len(self.data)
        try : 
            tId = self.rng.randint(0, n - 1)
            t = self.data[tId]
            ref = self.rng.sample(t.nodes, k=1).pop()
            # plus = self.rng.sample(list(t.nodes - [ref]), k=1).pop()
            plus = self.rng.sample(list(siblings(t, ref)), k=1).pop()
            refPlus = lcaScore(t, ref, plus) 
            # minusSet = [n for n in t.nodes if lcaScore(t, ref, n) > refPlus]
            # minus = self.rng.sample(minusSet, k=1).pop()
            minus = self.rng.sample(list(t.nodes - [ref] - siblings(t, ref)), k=1).pop()
        except Exception as e : 
            return self.getSample()
        refMinus = lcaScore(t, ref, minus)
        return (tId, ref, plus, minus, refPlus, refMinus)

    def __next__ (self) : 
        if self.i < len(self) : 
            self.i += 1
            return self.getSample()
        else :
            self.i = 0
            if self.val : 
                self.rng = random.Random(self.seed)
            raise StopIteration

    def __len__ (self) : 
        # Fixed number of samples for each epoch.
        return self.length

class TripletSVGDataSet (data.Dataset, Saveable) : 
    """
    Pre-processed trees from clustering algorithm.
    """
    def __init__ (self, pickleFileName, transform=None) : 
        super(TripletSVGDataSet, self).__init__() 
        with open(pickleFileName, 'rb') as fd : 
            self.svgDatas = pickle.load(fd) 
        mean = [0.8142, 0.8045, 0.7693]
        std = [0.3361, 0.3329, 0.3664]
        self.transform = T.Compose([
            torch.from_numpy,
            lambda t : t.float(),
            lambda t : t.permute((2, 0, 1)),
            lambda t : F.avg_pool2d (t, 2),
            T.Normalize(mean=mean, std=std)
        ])
        if transform is not None : 
            self.transform = T.Compose([transform, self.transform])

    def getNodeInput (self, tId, node) : 
        t = self.svgDatas[tId]
        im   = self.transform(t.image).unsqueeze(0)
        crop = self.transform(t.nodes[node]['crop']).unsqueeze(0)
        whole = self.transform(t.nodes[node]['whole']).unsqueeze(0)

        return dict(im=im, crop=crop, whole=whole)
        
    def __getitem__ (self, index) :
        tId, ref, plus, minus, refPlus, refMinus = index
        t = self.svgDatas[tId]
        im         = self.transform(t.image)
        refCrop    = self.transform(t.nodes[ref  ]['crop' ])
        refWhole   = self.transform(t.nodes[ref  ]['whole'])
        plusCrop   = self.transform(t.nodes[plus ]['crop' ])
        plusWhole  = self.transform(t.nodes[plus ]['whole'])
        minusCrop  = self.transform(t.nodes[minus]['crop' ])
        minusWhole = self.transform(t.nodes[minus]['whole'])
        return dict(
            im=im,
            refCrop=refCrop,
            refWhole=refWhole,
            plusCrop=plusCrop,
            plusWhole=plusWhole,
            minusCrop=minusCrop,
            minusWhole=minusWhole,
            refPlus=torch.tensor(refPlus),
            refMinus=torch.tensor(refMinus)
        )

if __name__ == "__main__" : 
    import json
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    generateData(commonConfig['train_directory'], 'train64.pkl')
    generateData(commonConfig['test_directory'], 'test64.pkl')
    generateData(commonConfig['cv_directory'], 'cv64.pkl')
    # data_ = TripletSVGDataSet('cv.pkl')
    # dataloader = data.DataLoader(data_, sampler=TripletSampler(data_.svgDatas, val=True), batch_size=10)
    # for e in range(2) :
    #     for batch in dataloader : 
    #         print(e)
