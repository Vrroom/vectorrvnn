import torch
from torch import nn
from vectorrvnn.utils import * 
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase
import numpy as np

class OneBranch (TripletBase) :
    """
    Process the entire image and the path subset 
    through convnets. Add positional encoding
    and use another MLP to combine them.
    """
    def __init__ (self, opts) :
        super(OneBranch, self).__init__(opts)
        self.vis.append(
            BBoxVisCallback(
                frequency=opts.frequency,
                env=opts.name + "_vis"
            )
        )
        self.conv = convBackbone(opts)

    def embedding (self, node, **kwargs) : 
        whole = node['whole']
        embed = self.conv(whole)
        return embed

    @classmethod
    def nodeFeatures (cls, t, ps, opts) : 
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
        )
        tensorApply(
            data,
            getTransform(opts),
            module=np
        )
        bbox = pathsetBox(t, ps)
        bbox = bbox / getDocBBox(t.doc)
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        return data


