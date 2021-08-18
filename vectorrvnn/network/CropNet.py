import torch 
from torch import nn
from torch.nn import functional as F
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase
import visdom

class CropNet (TripletBase) : 
    def __init__(self, opts):  
        super(CropNet, self).__init__(opts)
        self._api = visdom.Visdom(env=opts.name + "_output_norm")
        closeWindow(self._api, "cropnet-output-norm")
        self.i = 0
        self.vis.append(
            BBoxVisCallback(
                frequency=opts.frequency,
                env=opts.name + "_vis"
            )
        )
        self.bbox = nn.Sequential(
            nn.Linear(4, opts.hidden_size),
            nn.ReLU(),
            nn.Linear(opts.hidden_size, opts.embedding_size, bias=False)
        )
        self.ln1 = nn.LayerNorm(opts.embedding_size)
        self.ln2 = nn.LayerNorm(opts.embedding_size)
        # self.ln1.apply(getInitializer(opts))
        # self.ln2.apply(getInitializer(opts))
        self.bbox.apply(getInitializer(opts))
        self.crop = convBackbone(opts)

    def embedding (self, node, **kwargs) : 
        bbox = node['bbox']
        crop = node['crop']
        F_bbox = self.ln1(self.bbox(bbox))
        F_crop = self.ln2(self.crop(crop))
        joint = torch.cat((F_bbox, F_crop), dim=1)
        if self.i % 100 == 0 : 
            bboxNorm = F_bbox.norm(dim=1).mean().detach().cpu().item()
            cropNorm = F_crop.norm(dim=1).mean().detach().cpu().item()
            self._api.line([bboxNorm], [self.i], win="cropnet-output-norm", update="append", name="bbox")
            self._api.line([cropNorm], [self.i], win="cropnet-output-norm", update="append", name="crop")
        self.i += 1
        return joint

    @classmethod
    def nodeFeatures(cls, t, ps, opts) : 
        bbox = pathsetBox(t, ps)
        docbox = getDocBBox(t.doc)
        data = dict(
            im=rasterize(
                t.doc,
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
