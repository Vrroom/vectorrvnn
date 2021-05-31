import h5py
from PIL import Image
from tqdm import tqdm
from osTools import *
from more_itertools import flatten, chunked
import random
import warnings
# warnings.filterwarnings('error')
import numpy as np
import multiprocessing as mp
from skimage import transform
from functools import partial
from torch.utils import data

def readImage (fname, classNameMap) : 
    try : 
        with Image.open(fname) as im : 
            label = classNameMap[osp.split(osp.split(fname)[0])[1]]
            im64 = transform.resize(np.array(im), (64, 64))
            if im64.shape == (64, 64, 3) : 
                return im64, label
            else :
                return None, None
    except Exception : 
        return None, None

def convertImageFolder2H5 (dataDir, h5FileName) : 
    classNames = [osp.split(d)[1] for d in listdir(dataDir)]
    classNameMap = dict(zip(classNames, range(len(classNames))))
    dataPts = list(flatten(map(listdir, listdir(dataDir))))
    random.shuffle(dataPts)
    nChunks = 1000
    chunks = list(chunked(dataPts, len(dataPts) // nChunks))
    with h5py.File(h5FileName, 'w') as hf: 
        images = hf.create_dataset(
            name='images', 
            shape=(0, 64, 64, 3), 
            maxshape=(None, 64, 64, 3),
            compression="gzip", 
            compression_opts=9,
            chunks=(128, 64, 64, 3),
            dtype="float16"
        )
        labels = hf.create_dataset(
            name='labels', 
            shape=(0,),
            maxshape=(None,),
            compression="gzip",
            chunks=(128,),
            dtype="i8",
            compression_opts=9
        )
        for chunk in tqdm(chunks) : 
            with mp.Pool() as p : 
                data = list(p.map(partial(readImage, classNameMap=classNameMap), chunk))
            data = [d for d in data if d[0] is not None]
            ims  = np.stack([im for im, _ in data])
            labs = np.array([l for _, l in data])
            images.resize(images.shape[0] + ims.shape[0], axis=0)
            images[-ims.shape[0]:, :, :, :] = ims
            labels.resize(labels.shape[0] + labs.shape[0], axis=0)
            labels[-labs.shape[0]:] = labs

class Subset(data.Dataset):
    """
    Subset of a dataset at specified indices.

    Arguments:
    dataset (Dataset): The whole Dataset
    indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        im, labels = self.dataset[self.indices[idx]]
        im = im.astype(np.float32)
        return self.transform(image=im)['image'], labels

    def __len__(self):
        return len(self.indices)

class BamDataset(data.Dataset) : 
    def __init__ (self, h5FileName, transform=None) : 
        super(BamDataset, self).__init__()
        self.h5_file = h5py.File(h5FileName, 'r') 
        self.images = self.h5_file.get('images')
        self.labels = self.h5_file.get('labels')
        self.transform = transform

    def __getitem__ (self, index) : 
        im = self.images[index]
        label = self.labels[index]
        if self.transform is not None : 
            im = self.transform(im)
        return im, label

    def __len__ (self) : 
        return self.images.shape[0]

if __name__ == "__main__" : 
    BAM_DIR = '/misc/extra/data/sumitc/bam'
    BAM_H5 = '/misc/extra/data/sumitc/bam.h5'
    # if osp.exists(BAM_H5): 
    #     os.remove(BAM_H5)
    # convertImageFolder2H5(BAM_DIR, BAM_H5)
    data = BamDataset(BAM_H5)
    print(data[0])
