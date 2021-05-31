from functools import lru_cache
from matplotlib import colors
import xml.etree.ElementTree as ET
from copy import deepcopy
from .svgIO import GRAPHIC_TAGS

@lru_cache(maxsize=128)
def cachedPaths (doc) : 
    return doc.paths()

def pathColor (path, attr) : 
    if attr in path.element.attrib : 
        attr = colors.to_rgb(path.element.attrib[attr])
    else :
        attr = [1, 1, 1]
    return attr

def pathStrokeWidth (path) :
    if 'stroke-width' in path.element.attrib : 
        return float(path.element.attrib['stroke-width'])
    else :
        return 1

def fixOrigin (doc) : 
    ox, oy = doc.get_viewbox()[:2]
    globalTransform(doc, 
            dict(transform=f'translate({-ox} {-oy})'))

def scaleToFit (doc, h, w) : 
    ow, oh = doc.get_viewbox()[-2:]
    globalTransform(doc, 
            dict(transform=f'scale({w/ow} {h/oh})'))

def removeGraphicChildren (xmltree) :
    children = [e for e in xmltree if e.tag in GRAPHIC_TAGS]
    for child in children : 
        xmltree.remove(child)

def globalTransform(doc, transform) : 
    """ 
    Apply a global transform to the whole graphic
    by adding a group at the top most level. The
    transform is a dictionary specifying different 
    group attributes
    """
    groupElement = ET.Element('{http://www.w3.org/2000/svg}g', transform)
    rootCopy = deepcopy(doc.tree.getroot())
    # Don't add <defs ... /> to new group element.
    children = [e for e in rootCopy if e.tag in GRAPHIC_TAGS]
    groupElement.extend(children)
    root = doc.tree.getroot()
    # Remove graphic elements.
    removeGraphicChildren(root)
    root.append(groupElement)
    doc.tree._setroot(root)

def combineGraphics (doc1, doc2) : 
    pass


