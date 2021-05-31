# import subprocess
# import xml.etree.ElementTree as ET
# from copy import deepcopy
# import numpy as np
# import random
# import os
# import os.path as osp
# import svgpathtools as svg
# import matplotlib.image as image
# import string
import pathfinder_rasterizer as pr
from .svgTools import *

# def alphaComposite (source, module=np, color=[1,1,1]) : 
#     if module == np : 
#         color = np.array(color)
#         alpha = source[..., 3:] 
#         d_ = np.ones_like(source[..., :3])
#         d_[..., :] = color
#         s_ = source[..., :3]
#     else : 
#         color = torch.tensor(color)
#         alpha = source[:, 3:, ...] 
#         d_ = torch.ones_like(source[:, :3, ...])
#         d_[:, :, ...] = color
#         s_ = source[:, :3, ...]
#     return d_ * (1 - alpha) + s_ * alpha
 
def rasterize (doc, h, w) : 
    fixOrigin(doc)
    scaleToFit(doc, h, w)
    doc.set_viewbox(' '.join(map(str, [0, 0, h, w])))
    return pr.numpyRaster(doc) 

# def inheritedAttributes (doc, element) : 
#     attrs = deepcopy(element.attrib)
#     if element in doc.parent_map : 
#         parentElement = doc.parent_map[element]
#         parentAttrs = inheritedAttributes(doc, parentElement)
#         parentAttrs.update(attrs)
#         return parentAttrs
#     else :
#         return attrs
# 
# def getSubsetSvg(doc, paths, lst, vb) :
#     """
#     An svg is a collection of paths. 
#     This function chooses a list of
#     paths from the svg and makes an svg
#     string for those paths. 
# 
#     While doing this, we have to be careful
#     about the order in which these paths
#     are put because they determine the
#     order in which they are rendered. 
# 
#     For this, I have tweaked the 
#     svgpathtools library to also
#     store the zIndex of the paths
#     so that we have help while putting
#     the paths together
# 
#     >>> doc = svg.Document('file.svg')
#     >>> paths = doc.flatten_all_paths()
#     >>> print(getSubsetSvg(paths, [1,2,3], doc.get_viewbox()))
# 
#     Parameters
#     ----------
#     paths : list
#         List where each element is of
#         the type svg.Document.FlattenedPath
#     lst : list
#         An index set into the previous list 
#         specifying the paths we want.
#     vb : list
#         The viewbox of the original svg.
#     """
#     docCpy = deepcopy(doc)
#     vbox = ' '.join([str(_) for _ in vb])
#     docCpy.set_viewbox(vbox)
#     return ET.tostring(root, encoding='unicode')
# 
# def getSubsetSvg2(doc, paths, lst, vb) :
#     docCpy = deepcopy(doc)
#     vbox = ' '.join([str(_) for _ in vb])
#     docCpy.set_viewbox(vbox)
#     root = docCpy.root
#     unwantedPaths = list(set(range(len(paths))) - set(lst))
#     unwantedPathIds = [paths[i].zIndex for i in unwantedPaths]
#     allElts = list(root.iter())
#     unwantedElts = [allElts[i] for i in unwantedPathIds]
#     for elt in unwantedElts : 
#         docCpy.parent_map[elt].remove(elt)
#     return ET.tostring(root, encoding='unicode')
# 
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
# def SVGtoNumpyImage (svgFilePath, H, W, alpha=False) :
#     """
#     Take an SVG file and rasterize it to 
#     obtain a numpy array of given height and
#     width. 
# 
#     Parameters
#     ----------
#     svgFilePath : str
#     H : float
#         Desired height of output.
#     W : float 
#         Desired width of output.
#     """
#     with open(svgFilePath) as fd : 
#         string = fd.read()
#     return svgStringToBitmap(string, H, W, alpha)
