from functools import lru_cache
from matplotlib import colors
import xml.etree.ElementTree as ET
from copy import deepcopy
import svgpathtools as svg
from .svgIO import GRAPHIC_TAGS
import cssutils

@lru_cache(maxsize=128)
def cachedPaths (doc) : 
    return doc.paths()

def parseHex(s):
    s = s.lstrip('#')
    if len(s) == 3:
        s = s[0] + s[0] + s[1] + s[1] + s[2] + s[2]
    rgb = tuple(int(s[i:i+2], 16) for i in (0, 2, 4))
    return [rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]

def parseColor(s):
    s = s.lstrip(' ')
    color = [0, 0, 0]
    if s[0] == '#':
        color[:3] = parseHex(s)
    elif s == 'none':
        color = [0, 0, 0]
    elif s[:4] == 'rgb(':
        rgb = s[4:-1].split(',')
        color = [
            int(rgb[0]) / 255.0, 
            int(rgb[1]) / 255.0, 
            int(rgb[2]) / 255.0
        ]
    else:
        try : 
            color = colors.to_rgb(s)
        except ValueError : 
            warnings.warn('Unknown color command ' + s)
    return color

def pathColor (path, attr) : 
    if attr in path.element.attrib : 
        return parseColor(path.element.attrib[attr])
    else :
        if 'style' in path.element.attrib : 
            style = cssutils.parseStyle(
                    path.element.attrib['style'])
            if attr in style.keys() : 
                return parseColor(style.getPropertyValue(attr))
            else :
                return [1, 1, 1]
        else : 
            return [1, 1, 1]

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

def subsetSvg(doc, lst) :
    paths = cachedPaths(doc)
    nPaths = len(paths)
    docCpy = deepcopy(doc)
    root = docCpy.root
    unwantedPaths = list(set(range(nPaths)) - set(lst))
    unwantedPathIds = [paths[i].zIndex for i in unwantedPaths]
    allElts = list(root.iter())
    unwantedElts = [allElts[i] for i in unwantedPathIds]
    for elt in unwantedElts : 
        docCpy.parent_map[elt].remove(elt)
    newDocument = svg.Document(None)
    newDocument.fromString(ET.tostring(root, encoding='unicode'))
    return newDocument
