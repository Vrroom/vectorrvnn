from vectorrvnn.utils import *
from vectorrvnn.geometry import *
import matplotlib.image as image
from random import randint
import os
import os.path as osp
import svgpathtools as svg

def center_(bbox) :
    return bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2

def test_crop () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgFiles = [
        osp.join(chdir, 'data', '1F4A0.svg'),
        osp.join(chdir, 'data', '1F4A1.svg'),
        osp.join(chdir, 'data', '1F4A2.svg'),
        osp.join(chdir, 'data', '1F4A3.svg'),
        osp.join(chdir, 'data', '1F4A4.svg'),
    ]
    docs = list(map(svg.Document, svgFiles))
    crops = [crop(d, [0]) for d in docs]
    boxes = [c.get_viewbox() for c in crops]
    rasters = [rasterize(doc, box[2], box[3]) for doc, box in zip(crops, boxes)]
    for im, fname in zip(rasters, svgFiles) : 
        fullpath = osp.join('out', 'crop-' + getBaseName(fname) + '.png')
        image.imsave(fullpath, im)


def test_doc_transforms () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgFiles = [
        osp.join(chdir, 'data', '1F4A0.svg'),
        osp.join(chdir, 'data', '1F4A1.svg'),
        osp.join(chdir, 'data', '1F4A2.svg'),
        osp.join(chdir, 'data', '1F4A3.svg'),
        osp.join(chdir, 'data', '1F4A4.svg'),
    ]
    docs = list(map(svg.Document, svgFiles))
    # test rotation 
    bboxCenters = list(map(
        lambda x : center_(graphicBBox(x)),
        docs
    ))
    rotated = [
        rotate(
            doc, 
            degrees=randint(-180, 180), 
            px=c[0], py=c[1]
        ) 
        for doc, c in zip(docs, bboxCenters)
    ]
    rasters = [rasterize(doc, 200, 200) for doc in rotated]
    for im, fname in zip(rasters, svgFiles) : 
        fullpath = osp.join('out', 'rotated-' + getBaseName(fname) + '.png')
        image.imsave(fullpath, im)
    # test opacity 
    translucent = [
        modAttr(
            doc, 
            'fill-opacity',
            0.5
        )
        for doc in docs
    ]
    translucent = [
        modAttr(
            doc, 
            'stroke-opacity',
            0.5
        )
        for doc in translucent
    ]
    rasters = [rasterize(doc, 200, 200) for doc in translucent]
    for im, fname in zip(rasters, svgFiles) : 
        fullpath = osp.join('out', 'opacity-' + getBaseName(fname) + '.png')
        image.imsave(fullpath, im)
    # test stroke width
    strokeWidth = [
        modAttr(
            doc, 
            'stroke-width',
            10
        )
        for doc in docs
    ]
    rasters = [rasterize(doc, 200, 200) for doc in strokeWidth]
    for im, fname in zip(rasters, svgFiles) : 
        fullpath = osp.join('out', 'strokewidth-' + getBaseName(fname) + '.png')
        image.imsave(fullpath, im)
    # composition 
    composed = docUnion(modAttr(docs[0], 'fill-opacity', 0.7), docs[1])
    raster = rasterize(composed, 200, 200)
    fullpath = osp.join('out', 'composed' + '.png')
    image.imsave(fullpath, raster)
    assert True

test_crop()
