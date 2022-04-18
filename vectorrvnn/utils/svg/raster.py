from copy import deepcopy
import numpy as np
import torch
from .svgTools import *
from vectorrvnn.utils.boxes import *
from subprocess import call
from matplotlib.image import imread
import os

NO_GPU = call(['nvidia-smi']) > 0 

def alphaComposite (source, module=np, color=[1,1,1]) : 
    originalShape = source.shape
    if len(originalShape) < 4 : 
        source = source.reshape((1, *originalShape))
    if module == np : 
        source = np.transpose(source, (0, 3, 1, 2))
    alpha = source[:, 3:, ...] 
    d_ = module.ones_like(source[:, :3, ...])
    for i, c in enumerate(color) : 
        d_[:, i, ...] = c
    s_ = source[:, :3, ...]
    composited = d_ * (1 - alpha) + s_ * alpha
    if module == np : 
        composited = np.transpose(composited, (0, 2, 3, 1))
    if len(originalShape) < 4 : 
        composited = composited.squeeze()
    return composited

@immutable_doc
def rasterize (doc, w=None, h=None, threadLocal=False) : 
    """
    Rasterize a document to given height and width.
    Either both height and width are None or both are
    integers.
    """ 
    if NO_GPU : 
        return rasterizeInkscape(doc, w, h)
    import pathfinder_rasterizer as pr
    assert ((w is None and h is None) \
            or (w is not None and h is not None))
    fixOrigin(doc)
    if w is not None : 
        scaleToFit(doc, w, h)
    if threadLocal : 
        return pr.numpyRasterThreadLocal(doc)
    else : 
        return pr.numpyRaster(doc)

@immutable_doc 
def rasterizeInkscape (doc, w=None, h=None) :
    """
    Use inkscape as fallback to rasterize.
    """
    doc.save('tmp.svg')
    call(['inkscape', '--export-type=png', '-h', str(h), '-w', str(w), 'tmp.svg'])
    arr = imread('tmp.png')
    os.remove('tmp.svg')
    os.remove('tmp.png')
    return arr
