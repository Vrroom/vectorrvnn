import multiprocessing as mp
import vectorrvnn.trainutils.Constants as C
from vectorrvnn.utils import *
import torchvision.transforms as T

class TripletDataLoader () : 
    
    def __init__(self, batch_size, sampler, num_workers=mp.cpu_count()) :
        self.batch_size = batch_size
        self.sampler = sampler
        self.i = 0
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=C.mean, std=C.std),
        ])

    def __iter__ (self) :
        return self

    def _positions (self, t, node) : 
        paths = t.nodes[node]['pathSet']
        paddedPaths = (list(paths) + [-1] * (C.max_len - len(paths)))
        position = torch.tensor(paddedPaths, dtype=torch.long)
        return position

    def _nodefeatures (self, t, node) : 
        paths = t.nodes[node]['pathSet']
        # TODO: Add the normalize transform somewhere
        # Right now its ok because we are building it bit
        # by bit
        im = rasterize(subsetSvg(t.doc, paths), 256, 256)
        return self.transform(im), self._positions(t, node)

    def _tensorify (self, t, ref, plus, minus, refPlus, refMinus) : 
        refWhole  , refPositions   = self._nodefeatures(t, ref)
        plusWhole , plusPositions  = self._nodefeatures(t, plus)
        minusWhole, minusPositions = self._nodefeatures(t, minus)
        return dict(
            refWhole      =refWhole,
            refPositions  =refPositions,
            plusWhole     =plusWhole,
            plusPositions =plusPositions,
            minusWhole    =minusWhole,
            minusPositions=minusPositions,
            refPlus       =torch.tensor(refPlus),
            refMinus      =torch.tensor(refMinus)
        )
    
    def __next__ (self) : 
        if self.i + self.batch_size > len(self) :
            self.i = 0 
            self.sampler.reset()
            raise StopIteration
        else : 
            samples = [next(self.sampler) for _ in range(self.batch_size)]
            tensorified = [self._tensorify(*_) for _ in samples]
            return aggregateDict(tensorified, torch.stack)

    def __len__ (self) : 
        return len(self.sampler)



