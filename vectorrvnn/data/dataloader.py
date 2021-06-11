import multiprocessing as mp
from vectorrvnn.utils import *
import torchvision.transforms as T
from functools import partial

class TripletDataLoader () : 
    
    def __init__(self, opts, sampler) :
        self.opts = opts
        self.sampler = sampler
        self.i = 0
        transforms = [
            T.ToTensor(), 
            T.Lambda(lambda t : t.float()),
            T.Normalize(mean=opts.mean, std=opts.std)
        ]
        if opts.input_nc < 4 : 
            transforms.insert(
                1, 
                T.Lambda(
                    partial(alphaComposite, module=torch)
                )
            )
        self.transform = T.Compose(transforms)

    def __iter__ (self) :
        return self

    def _positions (self, t, node) : 
        paths = t.nodes[node]['pathSet']
        paddedPaths = (list(paths) + [-1] * (self.opts.max_len - len(paths)))
        position = torch.tensor(paddedPaths, dtype=torch.long)
        return position

    def _nodefeatures (self, t, node) : 
        paths = t.nodes[node]['pathSet']
        im = rasterize(
            subsetSvg(t.doc, paths), 
            self.opts.raster_size, 
            self.opts.raster_size
        )
        return self.transform(im), self._positions(t, node)

    def _tensorify (self, t, ref, plus, minus, refPlus, refMinus) : 
        im = self.transform(
            rasterize(
                t.doc, 
                self.opts.raster_size, 
                self.opts.raster_size
            )
        )
        refWhole  , refPositions   = self._nodefeatures(t, ref)
        plusWhole , plusPositions  = self._nodefeatures(t, plus)
        minusWhole, minusPositions = self._nodefeatures(t, minus)
        return dict(
            im            =im,
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
        if self.i >= len(self) :
            self.i = 0 
            self.sampler.reset()
            raise StopIteration
        else : 
            self.i += 1
            samples = [next(self.sampler) for _ in range(self.opts.batch_size)]
            tensorified = [self._tensorify(*_) for _ in samples]
            return aggregateDict(tensorified, torch.stack)

    def __len__ (self) : 
        return len(self.sampler) // self.opts.batch_size

