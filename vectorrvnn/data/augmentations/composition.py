""" Most code is stolen and reworked from albumentations """
from vectorrvnn.utils import *
from .transform import SVGDataTransform
import numpy as np
import random
from functools import reduce
from functional import compose

class Compose(SVGDataTransform):

    """Compose transforms.

    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying all list of transforms. Default: 1.0.
    """

    def __init__(self, transforms, p=1.0):
        super(Compose, self).__init__(p)
    
    def transform (self, svgdata, *args) : 
        return reduce(compose, transforms)(svgdata, *args)

class OneOf(SVGDataTransform):
    """Select one of transforms to apply.

    Transforms probabilities will be normalized to one 1, so in this case transforms probabilities works as weights.
    Args:
        transforms (list): list of transformations to compose.
        p (float): probability of applying selected transform. Default: 0.5.
    """

    def __init__(self, transforms, p=0.5):
        super(OneOf, self).__init__(p=p)
        transforms_ps = [t.p for t in transforms]
        s = sum(transforms_ps)
        self.transforms_ps = [t / s for t in transforms_ps]
        self.transforms = transforms

    def transform (self, svgdata, *args) :
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))
        t = random_state.choice(self.transforms, p=self.transforms_ps)
        return t(svgdata, *args)

