from .svgTools import *
from vectorrvnn.utils.boxes import * 
from copy import deepcopy
from functools import reduce

@immutable_doc
def translate (doc, tx, ty) : 
    globalTransform(doc, 
            dict(transform=f'translate({tx} {ty})'))
    return doc

@immutable_doc
def rotate (doc, degrees, px=0, py=0) :
    globalTransform(doc,
            dict(transform=f'rotate({degrees} {px} {py})'))
    return doc

@immutable_doc
def scale(doc, sx, sy=None) : 
    """ Modify this because it is more convenient to scale the document in place
    """
    if sy is None : 
        sy = sx
    globalTransform(doc,
            dict(transform=f'scale({sx} {sy})'))
    return doc

@immutable_doc
def modAttr (doc, attr, value) : 
    """ Modify attribute for all paths in document

    Examples : 
        1. modAttr(doc, 'opacity', 0.5) # set opacity to 0.5 
        2. modAttr(doc, 'stroke-width', 10) # set stroke-width to 10
        3. modAttr(doc, 'fill', "#ff0000") # set fill to red for all paths
        4. modAttr(doc, 'stroke', "#00ff00") # set stroke to blue for all paths
    """
    root = doc.tree.getroot()
    for elt in root.iter() : 
        if elt.tag in PATH_TAGS : 
            xmlAttributeSet(elt, attr, str(value))
    return doc

@immutable_doc
def modAttrs (doc, attrDict) :
    root = doc.tree.getroot()
    for elt in root.iter() : 
        if elt.tag in PATH_TAGS : 
            for k, v in attrDict.items():
                xmlAttributeSet(elt, k, str(v))
    return doc

@immutable_doc
def docUnion (doc, that) : 
    # add the canvases
    setDocBBox(doc, getDocBBox(doc) + getDocBBox(that))
    # add the paths
    thatRoot = deepcopy(that.tree.getroot())
    thatChildren = list(filter(
        lambda e : e.tag in GRAPHIC_TAGS, 
        thatRoot
    ))
    root = doc.tree.getroot()
    root.extend(thatChildren)
    return doc

@immutable_doc
def subsetSvg(doc, lst) :
    doc.updateParentMap()
    root = doc.tree.getroot()
    paths = list(filter(
        lambda e : e.tag in PATH_TAGS, 
        root.iter()
    ))
    n = len(paths)
    unwanted = list(set(range(n)) - set(lst))
    unwantedElts = [paths[i] for i in unwanted]
    for elt in unwantedElts : 
        doc.parent_map[elt].remove(elt)
    newDocument = svg.Document(None)
    newDocument.fromString(ET.tostring(root, encoding='unicode'))
    return newDocument

