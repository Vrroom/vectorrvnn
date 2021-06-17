""" Most code is stolen and reworked from albumentations """
from vectorrvnn.utils import *
import numpy as np
import random
from functools import reduce
from compose import compose

class DocTransform : 

    def __init__ (self, transform=None, p=1.0) : 
        if transform is None : 
            self.transform = lambda x : x
        else : 
            self.transform = transform
        self.p = p
    
    def __call__ (self, doc) : 
        toss = random.random()
        if toss < self.p :
            return self.transform(doc)
        else :
            return doc

class Compose(DocTransform):

    """Compose transforms.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying all list of transforms. Default: 1.0.
    """

    def __init__(self, transforms, p=1.0):
        transform = reduce(compose, transforms)
        super(Compose, self).__init__(transform, p)

class OneOf(DocTransform):
    """Select one of transforms to apply.

    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.
    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms, p=0.5):
        super(OneOf, self).__init__()
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]
        self.transforms = transforms

    def __call__(self, doc):
        if random.random() < self.p:
            random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
            t = random_state.choice(self.transforms.transforms, p=self.transforms_ps)
            return t(doc)
        else : 
            return doc

def getDocTransform(opts) : 
    return lambda *args: args[0]
