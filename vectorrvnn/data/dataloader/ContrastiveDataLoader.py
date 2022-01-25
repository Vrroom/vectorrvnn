from vectorrvnn.network import *
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .DataLoaderBase import *
from more_itertools import chunked, flatten

class ContrastiveDataLoader (DataLoaderBase) : 
    
    def getBatch (self) : 
        bsz, Bsz = self.opts.base_size, self.opts.batch_size
        csz = Bsz // bsz
        ts, ms, ps = next(self.sampler)
        ts = [self.psfeatures(*_) for _ in ts]
        ts = list(chunked(ts, bsz))
        ms = list(chunked(ms, bsz))
        ps = list(chunked(ps, bsz))
        batch = []
        for t, m, p in zip(ts, ms, ps) : 
            data = dict()
            data['nodes'] = aggregateDict(t, torch.stack)
            data['ms'] = list(map(torch.tensor, m))
            data['ps'] = list(map(torch.tensor, p))
            tensorApply(
                data,
                lambda t : t.to(self.opts.device)
            )
            batch.append(data)
        batch = aggregateDict(batch, list, [('nodes',), ('ps',), ('ms',)])
        batch['ps'] = list(flatten(batch['ps']))
        batch['ms'] = list(flatten(batch['ms']))
        return batch
