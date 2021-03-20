import os
import os.path as osp
import _pickle as cPickle
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
from more_itertools import unzip, chunked
from functools import lru_cache
from tqdm import tqdm
import h5py 
from PIL import Image

class Saveable () :
    def save (self, savePath) : 
        with open(savePath, 'wb') as fd :
            cPickle.dump(self, fd)

def tryTripletSVGData (i, j) :
    try : 
        return TripletSVGData(i, j)
    except Exception : 
        return None

def generateSuggeroData (dataDir, cPickleDir) : 
    dataPts = list(map(listdir, listdir(dataDir)))
    _, cPickles = unzip(dataPts)
    cPickles = list(cPickles)
    svgFiles =  []
    for f, _ in dataPts : 
        with open(f) as fp : 
            svgFiles.append(fp.read().strip())
    pts = list(zip(svgFiles, cPickles))
    chunkedPts = list(chunked(pts, len(pts) // 100))
    for i, chunk in enumerate(tqdm(chunkedPts)) : 
        with mp.Pool(maxtasksperchild=30) as p : 
            svgDatas = list(p.starmap(tryTripletSVGData, chunk, chunksize=12))
        cPickleFileName = osp.join(cPickleDir, f'{i}.pkl')
        with open(cPickleFileName, 'wb') as fd : 
            cPickle.dump(svgDatas, fd)

def generateAnnotatedData (dataDir, cPickleFileName) : 
    dataPts = map(listdir, listdir(dataDir))
    removeTxt = lambda x : filter(lambda y : not y.endswith('txt'), x)
    dataPts = list(map(lambda x : list(removeTxt(reversed(x))), dataPts))
    with mp.Pool(maxtasksperchild=30) as p : 
        svgDatas = list(p.starmap(tryTripletSVGData, dataPts))
    with open(cPickleFileName, 'wb') as fd : 
        cPickle.dump(svgDatas, fd)

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
            plus = self.rng.sample(list(siblings(t, ref)), k=1).pop()
            refPlus = lcaScore(t, ref, plus) 
            minusSet = t.nodes - [ref] - siblings(t, ref) - descendants(t, ref) - {parent(t, ref)}
            minus = self.rng.sample(list(minusSet), k=1).pop()
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

def lightBackgroundTransform (im) : 
    source = im[:3, :, :]
    alpha = im[3:, :, :]
    destination = torch.zeros_like(source)
    destination[0, :, :] = random.uniform(0.5, 1)
    destination[1, :, :] = random.uniform(0.5, 1)
    destination[2, :, :] = random.uniform(0.5, 1)
    return alpha * source + (1 - alpha) * destination

def whiteBackgroundTransform (im) : 
    source = im[:3, :, :]
    alpha = im[3:, :, :]
    destination = 0.7 * torch.ones_like(source)
    return alpha * source + (1 - alpha) * destination

class PickleDataset (data.Dataset) :

    def __init__ (self, cPickleDir, transform=None) :
        super(PickleDataset, self).__init__()
        self.files = listdir(cPickleDir)

    @lru_cache(maxsize=4)
    def __getitem__ (self, idx) :
        fname = self.files[idx]
        with open(fname, 'rb') as fp : 
            t = cPickle.load(fp)
        return t

    def __len__ (self) :
        return len(self.files)

class TripletSVGDataSet (data.Dataset, Saveable) : 
    """
    Pre-processed trees from clustering algorithm.
    """
    def __init__ (self, root, transform=None) : 
        super(TripletSVGDataSet, self).__init__() 
        self.files = listdir(root)
        self.svgDatas = []
        for f in tqdm(self.files) : 
            self.svgDatas.append(nx.read_gpickle(osp.join(f, 'tree.pkl')))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.transform = T.Compose([
            T.ToTensor(),
            whiteBackgroundTransform,
            T.Normalize(mean=mean, std=std)
        ])
        if transform is not None : 
            self.transform = T.Compose([transform, self.transform])

    def getNodeInput (self, tId, node) : 
        t = self.svgDatas[tId]
        im   = self.transform(t.nodes[findRoot(t)]['whole']).unsqueeze(0)
        crop = self.transform(t.nodes[node]['crop']).unsqueeze(0)
        whole = self.transform(t.nodes[node]['whole']).unsqueeze(0)
        return dict(im=im, crop=crop, whole=whole)

    def loadImage (self, fname, imageType, node, fromFile=True) : 
        with open(osp.join(fname, imageType, f'{node}.png'), 'rb') as f : 
            img = Image.open(f)
            return img.convert('RGBA')
        
    def __getitem__ (self, index) :
        tId, ref, plus, minus, refPlus, refMinus = index
        fname = self.files[tId]
        t = self.svgDatas[tId]
        im         = self.transform(self.loadImage(fname, 'whole', findRoot(t)))
        refCrop    = self.transform(self.loadImage(fname, 'crop' , ref))
        refWhole   = self.transform(self.loadImage(fname, 'whole', ref))
        plusCrop   = self.transform(self.loadImage(fname, 'crop' , plus))
        plusWhole  = self.transform(self.loadImage(fname, 'whole', plus))
        minusCrop  = self.transform(self.loadImage(fname, 'crop' , minus))
        minusWhole = self.transform(self.loadImage(fname, 'whole', minus))
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
    # generateSuggeroData('./unsupervised_v2', commonConfig['suggero_pickles'])
    generateAnnotatedData(commonConfig['train_directory'], 'train64.pkl')
    generateAnnotatedData(commonConfig['test_directory'], 'test64.pkl')
    generateAnnotatedData('ManuallyAnnotatedDataset_v2/Val', 'cv64.pkl')
