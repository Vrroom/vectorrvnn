from .UNet import * 
from .TwoBranch import convnet
from .TripletBase import TripletBase
from vectorrvnn.trainutils import *

class PatternGrouping (TripletBase) : 
    """ Architecture inspired by the pattern grouping paper """

    def __init__ (self, opts) : 
        super(PatternGrouping, self).__init__(opts)
        self.unet = UNet(opts)
        self.conv = convnet(opts)
        self.last = nn.Linear(
            opts.embedding_size + opts.structure_embedding_size + 4,
            opts.embedding_size
        )
        self.unet.apply(getInitializer(opts))
        self.last.apply(getInitializer(opts))
    
    def embedding (self, node, **kwargs) : 
        im = node['im']
        whole = node['whole']
        bitmap = node['bitmap']
        # [B, embedding_size]
        f1 = self.conv(whole) 
        # use the bitmap to perform weighted avg
        # [B, 8]
        f2 = (self.unet(im) * bitmap).sum(dim=(2, 3)) / (bitmap.sum() + 1e-6)
        # [B, 4]
        f3 = node['bbox'] 
        cat = torch.cat((f1, f2, f3), dim=1) 
        return self.last(cat)

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
        bitmap = np.copy(
            np.expand_dims(
                data['im'][:, :, -1], 
                0
            )
        )
        tensorApply(
            data,
            getTransform(opts),
            module=np
        )
        data['bitmap'] = torch.from_numpy(bitmap).float()
        bboxes = [t.nodes[i]['bbox'] for i in ps]
        bbox = reduce(lambda x, y : x + y, bboxes)
        data['bbox'] = torch.tensor(bbox.tolist())
        return data
