import os
import os.path as osp
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
from more_itertools import unzip, chunked
from tqdm import tqdm
import h5py 

class Saveable () :
    def save (self, savePath) : 
        with open(savePath, 'wb') as fd :
            pickle.dump(self, fd)

def tryTripletSVGData (i, j) :
    try : 
        return TripletSVGData(i, j)
    except Exception : 
        return None

def generateSuggeroData (dataDir, pickleDir) : 
    dataPts = list(map(listdir, listdir(dataDir)))
    _, pickles = unzip(dataPts)
    pickles = list(pickles)
    svgFiles =  []
    for f, _ in dataPts : 
        with open(f) as fp : 
            svgFiles.append(fp.read().strip())
    pts = list(zip(svgFiles, pickles))
    chunkedPts = list(chunked(pts, len(pts) // 100))
    for i, chunk in enumerate(tqdm(chunkedPts)) : 
        with mp.Pool(maxtasksperchild=30) as p : 
            svgDatas = list(p.starmap(tryTripletSVGData, chunk, chunksize=4))
        pickleFileName = osp.join(pickleDir, f'{i}.pkl')
        with open(pickleFileName, 'wb') as fd : 
            pickle.dump(svgDatas, fd)

def generateAnnotatedData (dataDir, pickleFileName) : 
    dataPts = map(listdir, listdir(dataDir))
    removeTxt = lambda x : filter(lambda y : not y.endswith('txt'), x)
    dataPts = list(map(lambda x : list(removeTxt(reversed(x))), dataPts))
    with mp.Pool(maxtasksperchild=30) as p : 
        svgDatas = list(p.starmap(tryTripletSVGData, dataPts))
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
    source = im[:, :, :3]
    alpha = im[:, :, 3:]
    destination = np.zeros_like(source)
    destination[:, :, 0] = random.uniform(0.5, 1)
    destination[:, :, 1] = random.uniform(0.5, 1)
    destination[:, :, 2] = random.uniform(0.5, 1)
    return alpha * source + (1 - alpha) * destination

def whiteBackgroundTransform (im) : 
    source = im[:, :, :3]
    alpha = im[:, :, 3:]
    destination = 1 * np.ones_like(source)
    return alpha * source + (1 - alpha) * destination

class TripletSVGDataSet (data.Dataset, Saveable) : 
    """
    Pre-processed trees from clustering algorithm.
    """
    def __init__ (self, pickleDir, transform=None) : 
        super(TripletSVGDataSet, self).__init__() 
        with open(pickleDir, 'rb') as fd :
            self.svgDatas = pickle.load(fd)
        #files = listdir(pickleDir)
        #self.svgDatas = []
        #for f in tqdm(files) :
        #    with open(f, 'rb') as fd :
        #        data = pickle.load(fd)
        #    data = list(filter(lambda x : x is not None, data))
        #    for i, _ in enumerate(data) :
        #        del data[i].image
        #        del data[i].bigImage
        #    self.svgDatas.extend(data)
        mean = [0.17859975,0.16340605,0.12297418,0.35452954]
        std = [0.32942199,0.30115585,0.25773552,0.46831796]
        self.transform = T.Compose([
            torch.from_numpy,
            lambda t : t.float(),
            lambda t : t.permute((2, 0, 1)),
            # lambda t : F.avg_pool2d (t, 2),
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
        
    def __getitem__ (self, index) :
        tId, ref, plus, minus, refPlus, refMinus = index
        t = self.svgDatas[tId]
        im         = self.transform(t.nodes[findRoot(t)]['whole'])
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
    # generateSuggeroData('./unsupervised', commonConfig['suggero_pickles'])
    # generateAnnotatedData(commonConfig['train_directory'], 'train.pkl')
    # generateAnnotatedData(commonConfig['test_directory'], 'test.pkl')
    generateAnnotatedData(commonConfig['cv_directory'], 'cv.pkl')
