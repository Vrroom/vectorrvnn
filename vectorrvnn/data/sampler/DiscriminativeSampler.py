import random
from vectorrvnn.utils import *
from itertools import permutations, combinations, starmap
from functools import partial
from .Sampler import *
from copy import deepcopy 

class DiscriminativeSampler (Sampler):  
    """ Just a sample of trees.  """
    def getSample (self) : 
        #bs = self.opts.batch_size
        #ids = self.rng.sample(list(range(len(self.svgdatas))), k=bs)
        return [self.transform(deepcopy(d), self.svgdatas) for d in self.svgdatas]
