import torch 
from torch import nn
from torch.nn import functional as F
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase

class BBoxNet (TripletBase) : 
    def __init__(self, opts):  
        super(BBoxNet, self).__init__(opts)
        self.vis.append(
            BBoxVisCallback(
                frequency=opts.frequency,
                env=opts.name + "_vis"
            )
        )
        self.net = nn.Sequential(
            nn.Linear(4, opts.hidden_size),
            nn.ReLU(),
            nn.Linear(opts.hidden_size, opts.embedding_size)
        )
        self.net.apply(getInitializer(opts))

    def embedding (self, node, **kwargs) : 
        bbox = node['bbox']
        return self.net(bbox)

    @classmethod
    def nodeFeatures(cls, t, ps, opts) : 
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
        docbox = getDocBBox(t.doc)
        bbox = bbox / docbox
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        return data
