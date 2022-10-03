from .svgTools import *
from vectorrvnn.utils.boxes import * 
from copy import deepcopy
from functools import reduce
import svgpathtools as svg

def toPath (lines) : 
    return svg.Path(*lines)

def toLines (path) : 
    return path._segments

def rotate (tree, degrees, pt) :
    globalTransform(tree.doc,
            dict(transform=f'rotate({degrees:.3f} {pt.real:.3f} {pt.imag:.3f})'))
    for i, line in enumerate(tree.lines) : 
        tree.lines[i] = toLines(toPath(line).rotated(degrees, pt))
    normalize(tree)
    tree.recalculateBoxes()
    return tree 

def translate(tree, tx, ty)  : 
    globalTransform(tree.doc,
            dict(transform=f'translate({tx:.3f} {ty:.3f})'))
    for i, line in enumerate(tree.lines) : 
        tree.lines[i] = toLines(toPath(line).translated(complex(tx, ty)))
    normalize(tree)
    tree.recalculateBoxes()
    return tree

def scale(tree, sx, sy=None) : 
    """ Modify this because it is more convenient to scale the document in place """
    if sy is None : 
        sy = sx
    globalTransform(tree.doc,
            dict(transform=f'scale({sx:.3f} {sy:.3f})'))
    for j, _ in enumerate(tree.lines) :
        for i, line in enumerate(tree.lines[j]) : 
            st = complex(line.start.real / sx, line.start.imag / sy)
            en = complex(line.end.real / sx, line.end.imag / sy)
            tree.lines[j][i] = svg.Line(start=st, end=en)
    normalize(tree)
    tree.recalculateBoxes()
    return tree

def modAttr (tree, attr, transformer) : 
    """ Modify attribute for all paths in document

    Examples : 
        1. modAttr(tree, 'opacity', lambda k, x: '0.5') # set opacity to 0.5 
        2. modAttr(tree, 'stroke-width', lambda k, x : '10') # set stroke-width to 10
        3. modAttr(tree, 'fill', lambda k, x : '#ff0000') # set fill to red for all paths
    """
    root = tree.doc.tree.getroot()
    for elt in root.iter() : 
        if elt.tag in PATH_TAGS : 
            xmlAttributeSet(elt, attr, transformer(attr, elt))
    return tree

def modAttrs (tree, attrDict) :
    """ apply modifications to attributes simultaneously """
    root = tree.doc.tree.getroot()
    for elt in root.iter() : 
        if elt.tag in PATH_TAGS : 
            for k, v in attrDict.items():
                xmlAttributeSet(elt, k, v(k, elt))
    return tree

def normalize (tree) : 
    box = union([pathBBox(svg.Path(*lines)) for lines in tree.lines])
    setDocBBox(tree.doc, box.normalized() * 1.2)
