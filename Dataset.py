import pickle
import multiprocessing as mp
from functools import partial
from torchUtils import imageForResnet
from torch.utils import data
from osTools import listdir
import torch
from SVGData import SVGData
from treeOps import *
from itertools import starmap
from tqdm import tqdm
import h5py 

class Saveable () :
    def save (self, savePath) : 
        with open(savePath, 'wb') as fd :
            pickle.dump(self, fd)

def generateData (dataDir, h5FileName) : 
    # TODO : CHECK WHY RASTERIZATION PRODUCES SO MANY WHITES
    dataPts = map(listdir, listdir(dataDir))
    dataPts = map(lambda l : [_ for _ in l if not _.endswith('png')], dataPts)
    dataPts = list(map(lambda x : list(reversed(x)), dataPts))
    with mp.Pool(maxtasksperchild=30) as p : 
        svgDatas = p.starmap(partial(SVGData, graph=None, samples=None), dataPts)
    svgDatas = list(filter(lambda x : maxOutDegree(x) <= 5, svgDatas))
    with h5py.File(h5FileName, 'w') as hf: 
        for k, t in enumerate(tqdm(svgDatas)) : 
            setNodeDepths(t)
            for i, a in enumerate(t.nodes): 
                for j, b in enumerate(t.nodes): 
                    l = lca(t, a, b)
                    if l != a and l != b and i < j: 
                        im1 = imageForResnet(t.nodes[a]['image']).numpy()
                        im2 = imageForResnet(t.nodes[b]['image']).numpy()
                        im = np.vstack((im1, im2))
                        g  = hf.create_group(f'group_{k}_{i}_{j}')
                        d = max(t.nodes[a]['depth'], t.nodes[b]['depth'])
                        y = min(1, (d - t.nodes[l]['depth']) / 10)
                        g.create_dataset(
                            name=f'Im',
                            data=im,
                            shape=im.shape,
                            maxshape=im.shape,
                            compression="gzip",
                            compression_opts=9
                        )
                        g.create_dataset(
                            name=f'lca',
                            data=y, 
                            shape=(1,),
                            compression="gzip",
                            compression_opts=9)

class SVGDataSet (data.Dataset, Saveable) : 
    """
    Pre-processed trees from clustering algorithm.
    """
    def __init__ (self, h5FileName) : 
        super(SVGDataSet, self).__init__() 
        self.h5_file = h5py.File(h5FileName, 'r')
        self.keys = list(self.h5_file.keys())

    def __getitem__ (self, index) :
        group = self.h5_file.get(self.keys[index])
        im = torch.from_numpy(group.get('Im')[()]).float()
        im1, im2 = torch.chunk(im, 2)
        lca = torch.from_numpy(group.get('lca')[()]).float()
        return im1, im2, lca

    def __len__ (self) : 
        return len(self.keys)

if __name__ == "__main__" : 
    import json
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    generateData(commonConfig['train_directory'], '__train__.h5')
    generateData(commonConfig['test_directory'], '__test__.h5')
    generateData(commonConfig['cv_directory'], '__cv__.h5')
