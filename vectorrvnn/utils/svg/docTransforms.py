from .svgTools import *
from vectorrvnn.geometry.boxes import * 
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
    for path in doc.paths() : 
        pathAttributeSet(path, attr, str(value))
    return doc

@immutable_doc
def docUnion (doc, that) : 
    # add the canvases
    setDocBBox(getDocBBox(doc) + getDocBBox(that))
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
    paths = cachedPaths(doc)
    nPaths = len(paths)
    root = doc.root
    unwantedPaths = list(set(range(nPaths)) - set(lst))
    unwantedPathIds = [paths[i].zIndex for i in unwantedPaths]
    allElts = list(root.iter())
    unwantedElts = [allElts[i] for i in unwantedPathIds]
    for elt in unwantedElts : 
        doc.parent_map[elt].remove(elt)
    newDocument = svg.Document(None)
    newDocument.fromString(ET.tostring(root, encoding='unicode'))
    return newDocument

