# import xml.etree.ElementTree as ET
# from copy import deepcopy
# import random
# import os
# import os.path as osp
# import matplotlib.image as image
# import string
import numpy as np
import torch
import pathfinder_rasterizer as pr
from .svgTools import *

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
        doc.set_viewbox(' '.join(map(str, [0, 0, h, w])))
    return pr.numpyRaster(doc) 


# 
# def SVGSubset2NumpyImage (doc, pathSet, H, W, alpha=False) :
#     paths = cachedFlattenPaths(doc)
#     boxes = np.array([paths[i].path.bbox() for i in pathSet])
#     docBox = doc.get_viewbox()
#     docDim = min(docBox[2], docBox[3])
#     eps = docDim / 20
#     xm, xM = boxes[:,0].min(), boxes[:,1].max()
#     ym, yM = boxes[:,2].min(), boxes[:,3].max()
#     h, w = xM - xm, yM - ym
#     d = max(h, w)
#     if h > w : 
#         box = [xm, ym+w/2-d/2, d, d]
#     else : 
#         box = [xm+h/2-d/2, ym, d, d]
#     box[0] -= eps
#     box[1] -= eps
#     box[2] += 2 * eps
#     box[3] += 2 * eps
#     svgString = getSubsetSvg(doc, paths, pathSet, box)
#     return svgStringToBitmap(svgString, H, W, alpha)
# 
# def SVGSubset2NumpyImage2 (doc, pathSet, H, W, alpha=False) :
#     paths = cachedFlattenPaths(doc)
#     boxes = np.array([paths[i].path.bbox() for i in pathSet])
#     docBox = doc.get_viewbox()
#     svgString = getSubsetSvg2(doc, paths, pathSet, docBox)
#     return svgStringToBitmap(svgString, H, W, alpha)
# 
