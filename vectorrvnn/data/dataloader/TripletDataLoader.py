from vectorrvnn.network import *
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .DataLoaderBase import *

class TripletDataLoader (DataLoaderBase) : 
    

    def _tensorify (self, t, ref, plus, minus, refPlus, refMinus) : 
        ref = self.nodefeatures(t, ref)
        plus = self.nodefeatures(t, plus)
        minus = self.nodefeatures(t, minus)
        return dict(
            ref=ref,
            plus=plus,
            minus=minus,
        )
    
    def getBatch (self) : 
        bsz, Bsz = self.opts.base_size, self.opts.batch_size
        csz = Bsz // bsz
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

