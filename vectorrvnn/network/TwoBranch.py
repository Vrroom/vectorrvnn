import torch
from torch import nn
from torch.nn import functional as F
from vectorrvnn.utils import * 
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase
import numpy as np

class TwoBranch (TripletBase) :
    """
    Process the entire image and the path subset 
    through convnets. Add positional encoding
    and use another MLP to combine them.
    """
    def __init__ (self, opts) :
        super(TwoBranch, self).__init__(opts)
        self.wholeEmbedder = convBackbone(opts)
        self.cropEmbedder  = convBackbone(opts)

    def embedding (self, node, **kwargs) : 
        whole = unitNorm(self.wholeEmbedder(node['whole']))
        crop  = unitNorm(self.cropEmbedder(node['crop']))
        return torch.cat((whole, crop), dim=1)

    @classmethod
    def nodeFeatures (cls, t, ps, opts) : 
        bbox = pathsetBox(t, ps)
        docbox = getDocBBox(t.doc)
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
            crop=rasterize(
                crop(subsetSvg(t.doc, ps), bbox, docbox), 
                opts.raster_size, 
                opts.raster_size
            )
        )
        tensorApply(
            data,
            getTransform(opts),
            module=np
        )
        bbox = bbox / docbox
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        return data


