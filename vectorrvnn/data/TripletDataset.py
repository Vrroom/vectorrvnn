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
from osTools import *
import torch
from TripletSVGData import TripletSVGData
from treeOps import *
from itertools import starmap, product
from more_itertools import unzip, chunked
from functools import lru_cache
from tqdm import tqdm
import h5py 
from PIL import Image
import Constants as C
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Saveable () :
    def save (self, savePath) : 
        with open(savePath, 'wb') as fd :
            cPickle.dump(self, fd)

def tryTripletSVGData (i, j) :
    try : 
        return TripletSVGData(i, j)
    except Exception as e: 
        print(e)
        return None

def generateSuggeroData (dataDir, outDir) : 
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
        for data in svgDatas : 
            if data is not None : 
                data.write(outDir)

def generateAnnotatedData (dataDir, outDir) : 
    dataPts = map(listdir, listdir(dataDir))
    removeTxt = lambda x : filter(lambda y : not y.endswith('txt'), x)
    dataPts = list(map(lambda x : list(removeTxt(reversed(x))), dataPts))
    with mp.Pool(maxtasksperchild=30) as p : 
        svgDatas = list(p.starmap(tryTripletSVGData, dataPts))
    for data in svgDatas :
        if data is not None : 
            data.write(outDir)

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
            plus = self.rng.sample(t.nodes - [ref], k=1).pop()
            minus = self.rng.sample(t.nodes - [ref], k=1).pop()
            attempts = 0
            while lcaScore(t, ref, plus) == lcaScore(t, ref, minus) and attempts < 10:
                attempts += 1
                plus = self.rng.sample(t.nodes - [ref], k=1).pop()
                minus = self.rng.sample(t.nodes - [ref], k=1).pop()
            if attempts >= 10 : 
                return self.getSample()
            if lcaScore(t, ref, plus) > lcaScore(t, ref, minus) :
                minus, plus = plus, minus
        except Exception as e : 
            return self.getSample()
        refMinus = lcaScore(t, ref, minus)
        refPlus = lcaScore(t, ref, plus)
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

def whiteBackgroundTransform (image) : 
    source = image[:, :, :3]
    alpha = image[:, :, 3:]
    destination = image.max() * np.ones_like(source)
    return alpha * source + (1 - alpha) * destination

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
            
        self.transform = A.Compose([
            A.Normalize(mean=C.mean, std=C.std, max_pixel_value=1),
            ToTensorV2()
        ])
        if transform is not None : 
            self.transform = A.Compose([transform, self.transform])

    def loadImage (self, fname, imageType, node, fromFile=True) : 
        with open(osp.join(fname, imageType, f'{node}.png'), 'rb') as f : 
            img = Image.open(f)
            img = np.array(img.convert('RGBA'))
            if img.dtype == np.uint8 : 
                img = img.astype(np.float) / 255
            assert img.max() < 1.1, "Image still in uint8"
            return whiteBackgroundTransform(img)
        
    def positions (self, tId, node) : 
        t = self.svgDatas[tId]
        paths = t.nodes[node]['pathSet']
        paddedPaths = (list(paths) + [-1] * (C.max_len - len(paths)))
        position = torch.tensor(paddedPaths, dtype=torch.long)
        return position
    
    def _getNodeFeatures (self, tId, node) :
        fname = self.files[tId]
        t = self.svgDatas[tId]
        crop     = self.transform(image=self.loadImage(fname, 'crop', node))['image']
        whole    = self.transform(image=self.loadImage(fname, 'whole', node))['image']
        position = self.positions(tId, node)
        return crop, whole, position
        
    def __getitem__ (self, index) :
        tId, ref, plus, minus, refPlus, refMinus = index
        fname = self.files[tId]
        t = self.svgDatas[tId]
        im         = self.transform(image=self.loadImage(fname, 'whole', findRoot(t)))['image']
        refCrop  , refWhole  , refPositions   = self._getNodeFeatures(tId, ref)
        plusCrop , plusWhole , plusPositions  = self._getNodeFeatures(tId, plus)
        minusCrop, minusWhole, minusPositions = self._getNodeFeatures(tId, minus)
        return dict(
            im            =im,
            refCrop       =refCrop,
            refWhole      =refWhole,
            refPositions  =refPositions,
            plusCrop      =plusCrop,
            plusWhole     =plusWhole,
            plusPositions =plusPositions,
            minusCrop     =minusCrop,
            minusWhole    =minusWhole,
            minusPositions=minusPositions,
            refPlus       =torch.tensor(refPlus),
            refMinus      =torch.tensor(refMinus)
        )

if __name__ == "__main__" : 
    import json
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # generateSuggeroData('./unsupervised_v2', commonConfig['suggero_dest'])
    generateAnnotatedData(commonConfig['val_directory'], commonConfig['val_dest'])
    generateAnnotatedData(commonConfig['test_directory'], commonConfig['test_dest'])
    generateAnnotatedData(commonConfig['train_directory'], commonConfig['train_dest'])
