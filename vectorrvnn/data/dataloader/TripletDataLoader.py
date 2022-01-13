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
        ref = self._nodefeatures(t, ref)
        plus = self._nodefeatures(t, plus)
        minus = self._nodefeatures(t, minus)
        return dict(
            ref=ref,
            plus=plus,
            minus=minus,
        )
    
    def __next__ (self) : 
        bsz, Bsz = self.opts.base_size, self.opts.batch_size
        csz = Bsz // bsz
        if self.i >= len(self) :
            self.reset()
            raise StopIteration
        else : 
            self.i += 1
            batch = []
            for i in range(csz) : 
                samples = [next(self.sampler) for _ in range(bsz)]
                tensorified = [self._tensorify(*_) for _ in samples]
                data = aggregateDict(tensorified, torch.stack)
                tensorApply(
                    data,
                    lambda t : t.to(self.opts.device)
                )
                batch.append(data)
            return aggregateDict(batch, list, [('ref',), ('plus',), ('minus',)])

    def reset (self) : 
        self.i = 0
        self.sampler.reset()

    def __len__ (self) : 
        return len(self.sampler) // self.opts.batch_size

