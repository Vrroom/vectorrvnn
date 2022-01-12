from vectorrvnn.network import *
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from more_itertools import chunked
from functools import partial
import numpy as np

class ContrastiveDataLoader () : 
    
    def __init__(self, opts, sampler) :
        self.opts = opts
        self.sampler = sampler
        self.i = 0
        self.transform = getTransform(opts)

    def __iter__ (self) :
        return self

    def _psFeature (self, t, ps) : 
        Cls = globals()[self.opts.modelcls]
        features = Cls.nodeFeatures(t, ps, self.opts)
        return features
    
    def __next__ (self) : 
        bsz, Bsz = self.opts.base_size, self.opts.batch_size
        csz = Bsz // bsz
        if self.i >= len(self) :
            self.reset()
            raise StopIteration
        else : 
            self.i += 1
            ts, ms, ps = next(self.sampler)
            ts = [self._psFeature(*_) for _ in ts]
            ts = list(chunked(ts, csz))
            ms = list(chunked(ms, csz))
            ps = list(chunked(ps, csz))
            batch = []
            for t, m, p in zip(ts, ms, ps) : 
                data = dict()
                data['nodes'] = aggregateDict(t, torch.stack)
                data['ms'] = torch.tensor(m) 
                data['ps'] = torch.tensor(p)
                tensorApply(
                    data,
                    lambda t : t.to(self.opts.device)
                )
                batch.append(data)
            return batch

    def reset (self) : 
        self.i = 0
        self.sampler.reset()

    def __len__ (self) : 
        return len(self.sampler) 
