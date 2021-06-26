from .svgTools import *
from vectorrvnn.utils.boxes import * 
from copy import deepcopy
from functools import reduce

def rotate (tree, degrees, pt) :
    globalTransform(tree.doc,
            dict(transform=f'rotate({degrees} {pt.real} {pt.imag})'))
    tree.recalculateBBoxes(lambda b : b.rotated(degrees, pt))
    normalize(tree)
    return tree

def scale(tree, sx, sy=None) : 
    """ Modify this because it is more convenient to scale the document in place
    """
    if sy is None : 
        sy = sx
    globalTransform(tree.doc,
            dict(transform=f'scale({sx} {sy})'))
    tree.recalculateBBoxes(lambda b : b.scaled(sx, sy))
    normalize(tree)
    return tree

def modAttr (tree, attr, value) : 
    """ Modify attribute for all paths in document

    Examples : 
        1. modAttr(tree, 'opacity', 0.5) # set opacity to 0.5 
        2. modAttr(tree, 'stroke-width', 10) # set stroke-width to 10
        3. modAttr(tree, 'fill', "#ff0000") # set fill to red for all paths
        4. modAttr(tree, 'stroke', "#00ff00") # set stroke to blue for all paths
    """
    root = tree.doc.tree.getroot()
    for elt in root.iter() : 
        if elt.tag in PATH_TAGS : 
            xmlAttributeSet(elt, attr, str(value))
    return tree

def modAttrs (tree, attrDict) :
    """ apply modifications to attributes simultaneously """
    root = tree.doc.tree.getroot()
    for elt in root.iter() : 
        if elt.tag in PATH_TAGS : 
            for k, v in attrDict.items():
                xmlAttributeSet(elt, k, str(v))
    return tree

def normalize (tree) : 
    rootBox = tree.nodes[findRoot(tree)]['bbox']
    setDocBBox(tree.doc, rootBox.normalized() * 1.2)
