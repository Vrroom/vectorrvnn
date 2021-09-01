import torch
from torch import nn
from vectorrvnn.utils import * 
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase
import numpy as np
from .RoIAlignNet import * 

class RoIBranch (TripletBase) :
    """
    Process the entire image and the path subset 
    through convnets. Add positional encoding
    and use another MLP to combine them.
    """
    def __init__ (self, opts) :
        super(RoIBranch, self).__init__(opts)
        self.vis.append(
            BBoxVisCallback(
                frequency=opts.frequency,
                env=opts.name + "_vis"
            )
        )
        self.roi = RoIAlignNet(opts)

    def embedding (self, node, **kwargs) : 
        im   = node['im']
        bbox = node['bbox_']
        return self.roi(im, bbox)

    @classmethod
    def nodeFeatures (cls, t, ps, opts) : 
        data = dict(
            im=rasterize(
                t.doc,
                opts.raster_size,
                opts.raster_size,
                opts.rasterize_thread_local
            )
        )
        tensorApply(
            data,
            getTransform(opts),
            module=np
        )
        bbox = pathsetBox(t, ps)
        bbox = bbox / getDocBBox(t.doc)
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        data['bbox_'] = torch.tensor(bbox.tolist(alternate=True)).float()
        return data

