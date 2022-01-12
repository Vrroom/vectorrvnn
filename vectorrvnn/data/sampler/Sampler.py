import random
from vectorrvnn.utils import *

class Sampler () : 
    """ 
    Generic Sampler 

    Samplers API -
        1. Iterators API - __len__, __iter__, __next__
        2. reset() - reset the count and the random number 
                generator seed if needed.
    """

    def __init__ (self, svgdatas, length, opts, 
            transform=lambda *args: args[0], val=False) :
        self.seed = rng.randint(0, 10000)
        self.rng = random.Random(self.seed) 
        self.svgdatas = svgdatas
        self.transform = transform 
        self.opts = opts
        # In val mode the same batches are sampled in each epoch.
        self.val = val 
        self.length = length
        self.i = 0

    def getSample (self) : 
        raise NotImplementedError

    def __iter__ (self): 
        return self

    def __next__ (self) : 
        if self.i < len(self) : 
            self.i += 1
            return self.getSample()
        else :
            self.i = 0
            if self.val : 
                self.rng = random.Random(self.seed)
            raise StopIteration

    def reset (self) : 
        self.i = 0
        if self.val : 
            self.rng = random.Random(self.seed)

    def __len__ (self) : 
        # Fixed number of samples for each epoch.
        return self.length

