import torch
from torch import nn
from .UNet import * 
from .TripletBase import TripletBase
from vectorrvnn.trainutils import *
from vectorrvnn.utils import *
from torchvision.models import *

def convnet (opts) : 
    model = resnet18(pretrained=True)
    model.fc = nn.Identity()
    model.float()
    return model

class PatternGroupingV2 (TripletBase) : 
    """ Architecture inspired by the pattern grouping paper """

    def __init__ (self, opts) : 
        super(PatternGroupingV2, self).__init__(opts)
        self.vis.append(
            BBoxVisCallback(
                frequency=opts.frequency,
                env=opts.name + "_vis"
            )
        )
        self.unet = UNet(opts)
        self.conv = convnet(opts)
        self.merger = nn.Linear(512 * 4, opts.embedding_size)
        self.last = nn.Linear(
            opts.embedding_size + opts.structure_embedding_size + 4,
            opts.embedding_size
        )
        self.unet.apply(getInitializer(opts))
        self.last.apply(getInitializer(opts))
    
    def embedding (self, node, **kwargs) : 
        im = node['im']
        crop1_2 = node['crop1_2']
        crop2_4 = node['crop2_4']
        crop3_6 = node['crop3_6']
        whole = node['whole']
        bitmap = node['bitmap']
        # [B, embedding_size]
        i1 = self.conv(whole) 
        i2 = self.conv(crop1_2)
        i3 = self.conv(crop2_4)
        i4 = self.conv(crop3_6)
        f1 = self.merger(torch.cat((i1, i2, i3, i4), 1))
        # use the bitmap to perform weighted avg
        # [B, 8]
        f2 = (self.unet(im) * bitmap).sum(dim=(2, 3)) / (bitmap.sum() + 1e-6)
        # [B, 4]
        f3 = node['bbox'] 
        cat = torch.cat((f1, f2, f3), dim=1) 
        return self.last(cat)

    @classmethod 
    def nodeFeatures (cls, t, ps, opts) : 
        bbox = pathsetBox(t, ps)
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
            crop1_2=rasterize(
                crop(t.doc, bbox * 1.2),
                opts.raster_size, 
                opts.raster_size,
            ),
            crop2_4=rasterize(
                crop(t.doc, bbox * 2.4),
                opts.raster_size, 
                opts.raster_size,
            ),
            crop3_6=rasterize(
                crop(t.doc, bbox * 3.6),
                opts.raster_size,
                opts.raster_size
            )
        )
        bitmap = np.copy(
            np.expand_dims(
                data['whole'][:, :, -1], 
                0
            )
        )
        tensorApply(
            data,
            getTransform(opts),
            module=np
        )
        data['bitmap'] = torch.from_numpy(bitmap).float()
        # Normalize by document's bbox
        bbox = bbox / getDocBBox(t.doc)
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        return data


