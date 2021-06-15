import torch
from torch import nn
from vectorrvnn.utils import * 
from vectorrvnn.trainutils import *
from .PositionalEncoding import PositionalEncoding
from .TripletBase import TripletBase
from torchvision.models import *
import numpy as np

def convnet (opts) : 
    model = resnet18(pretrained=True)
    # Use the weights of the pretrained model to 
    # create weights for the new model.
    inFeatures = model.fc.in_features
    model.fc = nn.Linear(inFeatures, opts.embedding_size)
    # Make sure that parameters are floats 
    model = model.float()
    return model

class OneBranch (TripletBase) :
    """
    Process the entire image and the path subset 
    through convnets. Add positional encoding
    and use another MLP to combine them.
    """
    def __init__ (self, opts) :
        super(OneBranch, self).__init__(opts)
        self.conv = convnet(opts)
        self.pe = PositionalEncoding(opts)

    def embedding (self, node, **kwargs) : 
        whole = node['whole']
        position = node['position']
        embed = self.pe(self.conv(whole), position)
        return embed

    @classmethod
    def nodeFeatures (cls, t, ps, opts) : 
        max_len = opts.max_len
        data = dict(
            whole=rasterize(
                subsetSvg(t.doc, ps),
                opts.raster_size,
                opts.raster_size
            ),
            position=np.array(list(ps) + [-1] * (max_len - len(ps)))
        )
        tensorApply(
            data,
            getTransform(opts),
            partial(isImage, module=np),
            module=np
        )
        tensorApply(
            data, 
            torch.from_numpy, 
            lambda x : not isImage(x, module=np),
            module=np
        )
        return data


