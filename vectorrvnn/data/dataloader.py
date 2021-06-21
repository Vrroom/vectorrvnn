from vectorrvnn.network import *
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from functools import partial
import numpy as np

class TripletDataLoader () : 
    
    def __init__(self, opts, sampler) :
        self.opts = opts
        self.sampler = sampler
        self.i = 0
        self.transform = getTransform(opts)

    def __iter__ (self) :
        return self

    def _nodefeatures (self, t, node) : 
        paths = t.nodes[node]['pathSet']
        Cls = globals()[self.opts.modelcls]
        features = Cls.nodeFeatures(t, paths, self.opts)
        return features

    def _tensorify (self, t, ref, plus, minus, refPlus, refMinus) : 
        im = self.transform(
            rasterize(
                t.doc, 
                self.opts.raster_size, 
                self.opts.raster_size
            )
        )
        ref = self._nodefeatures(t, ref)
        plus = self._nodefeatures(t, plus)
        minus = self._nodefeatures(t, minus)
        return dict(
            im=im,
            ref=ref,
            plus=plus,
            minus=minus,
        )
    
    def __next__ (self) : 
        if self.i >= len(self) :
            self.reset()
            raise StopIteration
        else : 
            self.i += 1
            samples = [next(self.sampler) for _ in range(self.opts.batch_size)]
            tensorified = [self._tensorify(*_) for _ in samples]
            return aggregateDict(tensorified, torch.stack)

    def reset (self) : 
        self.i = 0
        self.sampler.reset()

    def __len__ (self) : 
        return len(self.sampler) // self.opts.batch_size

