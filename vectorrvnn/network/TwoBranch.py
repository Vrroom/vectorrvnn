import torch
from torch import nn
from vectorrvnn.utils import * 
from vectorrvnn.trainutils import *
from .PositionalEncoding import PositionalEncoding
from .TripletBase import TripletBase
import numpy as np
from torchvision.models import *

def convnet (opts) : 
    model = resnet18(pretrained=True)
    # Use the weights of the pretrained model to 
    # create weights for the new model.
    inFeatures = model.fc.in_features
    model.fc = nn.Linear(inFeatures, opts.embedding_size)
    # Make sure that parameters are floats 
    model = model.float()
    return model

class TwoBranch (TripletBase) :
    """
    Process the entire image and the path subset 
    through convnets. Add positional encoding
    and use another MLP to combine them.
    """
    def __init__ (self, opts) :
        super(TwoBranch, self).__init__(opts)
        self.conv = convnet(opts)
        self.pe = PositionalEncoding(opts)
        if opts.hidden_size is not None : 
            self.nn = nn.Sequential(
                nn.Linear(2 * opts.embedding_size, opts.hidden_size),
                nn.ReLU(),
                nn.Linear(opts.hidden_size, opts.embedding_size)
            )
        else : 
            self.nn = nn.Linear(
                2 * opts.embedding_size, 
                opts.embedding_size
            )

    def embedding (self, node, **kwargs) : 
        im = node['im']
        whole = node['whole']
        position = node['position']
        imEmbed = self.conv(im)
        wholeEmbed = self.pe(self.conv(whole), position)
        cat = torch.cat((imEmbed, wholeEmbed), dim=1)
        return self.nn(cat)

    @classmethod
    def nodeFeatures (cls, t, ps, opts) : 
        max_len = opts.max_len
        data = dict(
            im=rasterize(
                t.doc,
                opts.raster_size,
                opts.raster_size
            ),
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


