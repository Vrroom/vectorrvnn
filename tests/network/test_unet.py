from vectorrvnn.network import *
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from more_itertools import collapse
import torchvision.transforms as T
import torch
import os
import os.path as osp

def _test_unet () :
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot', 
        osp.join(chdir, '../../data/Toy'),
        '--checkpoints_dir', 
        osp.join(chdir, '../../results'),
        '--name', 
        'test_unet'
    ])
    unet = UNet(opts).to(opts.device).float()
    valDir = osp.join(opts.dataroot, 'Val')
    svgFile = next(filter(
        lambda x : x.endswith('svg'), 
        allfiles(valDir)
    ))
    doc = svg.Document(svgFile)
    im = alphaComposite(rasterize(doc, opts.raster_size, opts.raster_size))
    im = T.ToTensor()(im).to(opts.device).float().unsqueeze(0)
    out = unet(im)
    assert(out.shape[1] == 8)
    assert(out.shape[2:] == im.shape[2:])

