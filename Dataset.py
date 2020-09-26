import pickle
import multiprocessing as mp
from functools import partial
from torch.utils import data
from osTools import listdir
from SVGData import SVGData

class Saveable () :
    def save (self, savePath) : 
        with open(savePath, 'wb') as fd :
            pickle.dump(self, fd)

class SVGDataSet (data.Dataset, Saveable) : 
    """
    Pre-processed trees from clustering algorithm.
    """
    def __init__ (self, svgDir, graph, samples, **kwargs) : 
        self.svgDir = svgDir
        self.svgFiles = listdir(svgDir) 
        with mp.Pool(maxtasksperchild=30) as p : 
            self.svgDatas = p.map(partial(SVGData, graph=graph, samples=samples), self.svgFiles)

    def __getitem__ (self, index) :
        return self.svgDatas[index]

    def toTensor (self) : 
        for i in range(len(self)) : 
            self.svgDatas[i].toTensor()

    def __len__ (self) : 
        return len(self.svgFiles)

class DatasetCache (Saveable) :

    def __init__ (self, svgDir) :
        self.svgDir = svgDir 
        self.cache = dict()

    def dataset (config) : 
        graph = config['graph'] 
        samples = config['samples']
        key = str((graph, samples))
        if key not in self.cache : 
            self.cache[key] = SVGDataSet(self.svgDir, graph, samples)
        return self.cache[key]

if __name__ == "__main__" : 
    dataset = SVGDataSet('/Users/amaltaas/BTP/vectorrvnn/PartNetSubset/CV', 'adjGraph', 10)
    dataset.toTensor() 
    print(dataset[0].descriptors.shape)
