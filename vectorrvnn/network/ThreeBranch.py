import torch 
from torch import nn
from torch.nn import functional as F
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase
from .RoIAlignNet import * 
import visdom

class ThreeBranch (TripletBase) : 
    def __init__(self, opts):  
        super(ThreeBranch, self).__init__(opts)
        self._api = visdom.Visdom(env=opts.name + "_output_norm")
        closeWindow(self._api, "three-branch-output-norm")
        self.i = 0
        self.vis.append(
            BBoxVisCallback(
                frequency=opts.frequency,
                env=opts.name + "_vis"
            )
        )
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
        if self.i % 100 == 0 : 
            bboxNorm = F_bbox.norm(dim=1).mean().detach().cpu().item()
            cropNorm = F_crop.norm(dim=1).mean().detach().cpu().item()
            roiNorm  = F_roi.norm(dim=1).mean().detach().cpu().item()
            self._api.line([bboxNorm], [self.i], win="cropnet-output-norm", update="append", name="bbox")
            self._api.line([cropNorm], [self.i], win="cropnet-output-norm", update="append", name="crop")
            self._api.line([roiNorm], [self.i], win="cropnet-output-norm", update="append", name="roi")
        self.i += 1
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
        bbox = bbox / docbox
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        data['bbox_'] = torch.tensor(bbox.tolist(alternate=True)).float()
        return data

