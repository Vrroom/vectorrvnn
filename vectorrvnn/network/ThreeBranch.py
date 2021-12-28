import torch 
from torch import nn
from torch.nn import functional as F
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase
from .RoIAlignNet import * 

class ThreeBranch (TripletBase) : 
    def __init__(self, opts):  
        super(ThreeBranch, self).__init__(opts)
        self.bbox = fcn(opts, 4, opts.embedding_size)
        self.crop = convBackbone(opts)
        self.combine = fcn(opts, 3 * opts.embedding_size, opts.embedding_size)
        self.roi = RoIAlignNet(opts)

    def embedding (self, node, **kwargs) : 
        im    = node['im']
        bbox  = node['bbox']
        bbox_ = node['bbox_']
        crop  = node['crop']
        F_roi  = self.roi(im, bbox_)
        F_bbox = self.bbox(bbox)
        F_crop = self.crop(crop)
        joint = torch.cat((F_roi, F_bbox, F_crop), dim=1)
        return self.combine(joint)

    @classmethod
    def nodeFeatures(cls, t, ps, opts) : 
        bbox = pathsetBox(t, ps)
        docbox = getDocBBox(t.doc)
        data = dict(
            im=rasterize(
                t.doc,
                opts.raster_size,
                opts.raster_size,
                opts.rasterize_thread_local
            ),
            crop=rasterize(
                crop(subsetSvg(t.doc, ps), bbox, docbox), 
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
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        data['bbox_'] = torch.tensor(bbox.tolist(alternate=True)).float()
        return data

