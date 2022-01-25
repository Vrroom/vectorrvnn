from vectorrvnn.network import *
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .DataLoaderBase import *

class DiscriminativeDataLoader (DataLoaderBase) : 
    
    def getBatch (self) : 
        bs = self.opts.batch_size
        ts = next(self.sampler)
        return dict(ts=ts)

