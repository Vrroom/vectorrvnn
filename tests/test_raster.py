import pytest
from vectorrvnn.utils import *
import svgpathtools as svg
import xml.etree.ElementTree as ET
import matplotlib.image as image
import os
import os.path as osp

def test_raster () :
    chdir = osp.split(osp.abspath(__file__))[0]
    for f in listdir(osp.join(chdir, 'data')) : 
        fname = getBaseName(f)
        doc = svg.Document(f)
        im = rasterize(doc, 1000, 1000)
        image.imsave(osp.join(chdir, 'out', fname + '.png'), im)
        assert(im.min() != im.max())
