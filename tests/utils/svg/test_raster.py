import pytest
from vectorrvnn.utils import *
import svgpathtools as svg
import xml.etree.ElementTree as ET
import matplotlib.image as image
import os
import os.path as osp
import numpy as np
import torch

def fpAssert (v1, v2) : 
    assert(abs(v1 - v2) < 1e-4)

def test_raster () :
    chdir = osp.split(osp.abspath(__file__))[0]
    for f in listdir(osp.join(chdir, 'data')) : 
        fname = getBaseName(f)
        doc = svg.Document(f)
        im = rasterize(doc, 1000, 1000)
        image.imsave(osp.join(chdir, 'out', fname + '.png'), im)
        assert(im.min() != im.max())

def test_alphacomposite () : 
    # transparent black screen
    im1 = np.zeros((100, 100, 4))
    im2 = torch.zeros((10, 4, 100, 100))
    im3 = torch.zeros((4, 100, 100))

    a1 = alphaComposite(im1, np, [0.5, 0.3, 0.2])
    fpAssert(a1[:, :, 0].mean(), 0.5)
    fpAssert(a1[:, :, 1].mean(), 0.3)
    fpAssert(a1[:, :, 2].mean(), 0.2)

    a2 = alphaComposite(im2, torch) 
    a3 = alphaComposite(im3, torch)

    fpAssert(a2.mean(), 1.0)
    fpAssert(a3.mean(), 1.0)
