from vectorrvnn.network import *
from vectorrvnn.trainutils import *

class DataLoaderBase() : 

    def psfeatures (self, t, ps) : 
        Cls = globals()[self.opts.modelcls]
        features = Cls.nodeFeatures(t, ps, self.opts)
        return features
    

    def nodefeatures (self, t, node) : 
        paths = t.nodes[node]['pathSet']
        Cls = self.opts.modelcls
        features = Cls.nodeFeatures(t, paths, self.opts)
        return features

    def __init__(self, opts, sampler) :
        self.opts = opts
        self.sampler = sampler
        self.i = 0
        self.transform = getTransform(opts)

    def __iter__ (self) :
        return self

    def getBatch(self) : 
        raise NotImplementedError

    def __next__ (self) : 
        if self.i >= len(self) :
            self.reset()
            raise StopIteration
        else : 
            self.i += 1
            return self.getBatch()

    def reset (self) : 
        self.i = 0
        self.sampler.reset()

    def __len__ (self) : 
        return len(self.sampler) // self.opts.batch_size

