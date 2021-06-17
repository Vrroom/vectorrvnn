from vectorrvnn.utils import *
from vectorrvnn.geometry import *
import matplotlib.image as image
from random import randint
import os
import os.path as osp
import svgpathtools as svg

def test_doc_transforms () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgFiles = list(filter(
        lambda x : '1F4A' in x,
        listdir(osp.join(chdir, 'data'))
    ))
    docs = list(map(svg.Document, svgFiles))
    # test rotation 
    bboxes = list(map(getDocBBox, docs))
    bboxCenters = list(map(lambda x: x.center(), bboxes))
    rotated = [
        rotate(
            doc, 
            degrees=randint(-180, 180), 
            px=c.real, py=c.imag
        ) 
        for doc, c in zip(docs, bboxCenters)
    ]
    rasters = [rasterize(doc, 200, 200) for doc in rotated]
    for im, fname in zip(rasters, svgFiles) : 
        fullpath = osp.join(chdir, 'out', 'rotated-' + getBaseName(fname) + '.png')
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
        fullpath = osp.join(chdir, 'out', 'opacity-' + getBaseName(fname) + '.png')
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
        fullpath = osp.join(chdir, 'out', 'strokewidth-' + getBaseName(fname) + '.png')
        image.imsave(fullpath, im)
    # composition 
    composed = docUnion(modAttr(docs[0], 'fill-opacity', 0.7), docs[1])
    raster = rasterize(composed, 200, 200)
    fullpath = osp.join(chdir, 'out', 'composed' + '.png')
    image.imsave(fullpath, raster)
    assert True

