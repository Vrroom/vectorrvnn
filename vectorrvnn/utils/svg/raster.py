from copy import deepcopy
import numpy as np
import torch
import pathfinder_rasterizer as pr
from .svgTools import *
from vectorrvnn.geometry.boxes import *

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
def rasterize (doc, h=None, w=None) : 
    """
    Rasterize a document to given height and width.
    Either both height and width are None or both are
    integers.
    """ 
    assert ((h is None and w is None) \
            or (h is not None and w is not None))
    fixOrigin(doc)
    if h is not None : 
        scaleToFit(doc, h, w)
        setDocBBox(DimBBox(0, 0, w, h))
    return pr.numpyRaster(doc)
