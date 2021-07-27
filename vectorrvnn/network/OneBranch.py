import torch
from torch import nn
from vectorrvnn.utils import * 
from vectorrvnn.trainutils import *
from .PositionalEncoding import PositionalEncoding
from .TripletBase import TripletBase
from .TwoBranch import convnet
from torchvision.models import *
import numpy as np

class OneBranch (TripletBase) :
    """
    Process the entire image and the path subset 
    through convnets. Add positional encoding
    and use another MLP to combine them.
    """
    def __init__ (self, opts) :
        super(OneBranch, self).__init__(opts)
        self.conv = convnet(opts)

    def embedding (self, node, **kwargs) : 
        whole = node['whole']
        embed = self.conv(whole)
        return embed

    @classmethod
    def nodeFeatures (cls, t, ps, opts) : 
        data = dict(
            whole=rasterize(
                subsetSvg(t.doc, ps),
                opts.raster_size,
                opts.raster_size
            ),
        )
        tensorApply(
            data,
            getTransform(opts),
            module=np
        )
        return data


