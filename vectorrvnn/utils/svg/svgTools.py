from functools import lru_cache
import re
import string
from matplotlib import colors
import xml.etree.ElementTree as ET
from copy import deepcopy
import networkx as nx
import itertools, more_itertools
import svgpathtools as svg
from vectorrvnn.utils.graph import *

# Stroke attributes. 
STROKE_LINECAP = ['butt', 'round', 'square'] # default in butt
STROKE_LINEJOIN = ['miter', 'arcs', 'bevel', # default is miter
        'miter-clip', 'round'] 

# Defaults for path attributes
DEFAULT_COLOR = 'black'
DEFAULT_LINECAP = 'butt'
DEFAULT_LINEJOIN = 'miter'
DEFAULT_STROKEWIDTH = '1'

GRAPHIC_TAGS = [
    '{http://www.w3.org/2000/svg}rect',
    '{http://www.w3.org/2000/svg}circle',
    '{http://www.w3.org/2000/svg}ellipse',
    '{http://www.w3.org/2000/svg}line', 
    '{http://www.w3.org/2000/svg}polyline',
    '{http://www.w3.org/2000/svg}polygon',
    '{http://www.w3.org/2000/svg}path',
    '{http://www.w3.org/2000/svg}switch',
    '{http://www.w3.org/2000/svg}g',
]

class Del:
    def __init__(self, keep=string.digits + '.-e'):
        self.comp = dict((ord(c),c) for c in keep)
    def __getitem__(self, k):
        return self.comp.get(k)
DD = Del()

def getTreeStructureFromSVG (svgFile) : 
    """
    Infer the tree structure from the
    XML document. 

    Parameters
    ----------
    svgFile : str
        Path to the svgFile.
    """

    def buildTreeGraph (element) :
        """
        Recursively create networkx tree and 
        add path indices to the tree.

        Parameters
        ----------
        element : Element
            Element at this level of the
            tree.
        """
        nonlocal r
        curId = r
        r += 1
        T.add_node(curId)
        if element.tag in childTags : 
            zIdx = allNodes.index(element)
            pathIdx = zIndexMap[zIdx]
            T.nodes[curId]['pathSet'] = (pathIdx,)
        else : 
            childIdx = []
            validTags = lambda x : x.tag in childTags or x.tag == groupTag
            children = list(map(buildTreeGraph, filter(validTags, element)))
            T.add_edges_from(list(itertools.product([curId], children)))
            pathSet = tuple(more_itertools.collapse([T.nodes[c]['pathSet'] for c in children]))
            T.nodes[curId]['pathSet'] = pathSet
        return curId

    childTags = GRAPHIC_TAGS[:-1]
    groupTag = GRAPHIC_TAGS[-1]
    doc = svg.Document(svgFile)
    paths = doc.paths()
    zIndexMap = dict([(p.zIndex, i) for i, p in enumerate(paths)])
    tree = doc.tree
    root = tree.getroot()
    allNodes = list(root.iter())
    T = nx.DiGraph()
    r = 0
    buildTreeGraph (root)
    T = removeOneOutDegreeNodesFromTree(T)
    return T

@lru_cache(maxsize=128)
def cachedPaths (doc) : 
    return doc.paths()

def parseDashArray(da) : 
    return list(map(float, filter(
        lambda x : len(x) > 0, 
        re.split(',| ', da))))

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

def pathAttribute (path, attr, default) : 
    if attr in path.element.attrib : 
        return path.element.attrib[attr]
    elif 'style' in path.element.attrib : 
        style = path.element.attrib['style']
        style = dict(map(
                lambda x : x.split(':'), 
                style.split(';')))
        if attr in style : 
            return style[attr]
    return default
    
def pathColor (path, attr) : 
    """ attr can be one of stroke/fill """
    return parseColor(pathAttribute(path, attr, DEFAULT_COLOR))

def pathStrokeWidth (path) :
    pw = pathAttribute(path, 'stroke-width', DEFAULT_STROKEWIDTH)
    return float(pw.translate(DD))

def pathStrokeLineCap (path): 
    return pathAttribute(path, 'stroke-linecap', DEFAULT_LINECAP)

def pathStrokeLineJoin (path): 
    return pathAttribute(path, 'stroke-linejoin', DEFAULT_LINEJOIN)

def pathStrokeDashArray(path) : 
    """ 
    Instead of the whole dasharray string, just 
    say whether there is this attribute or not
    """
    da = pathAttribute(path, 'stroke-dasharray', None)
    return da if da != 'none' else None

def fixOrigin (doc) : 
    ox, oy = doc.get_viewbox()[:2]
    globalTransform(doc, 
            dict(transform=f'translate({-ox} {-oy})'))

def scaleToFit (doc, h, w) : 
    ow, oh = doc.get_viewbox()[-2:]
    globalTransform(doc, 
            dict(transform=f'scale({w/ow} {h/oh})'))

def translate (doc, tx, ty) : 
    globalTransform(doc, 
            dict(transform=f'translate({tx} {ty})'))

def rotate (doc, degrees, px=0, py=0) :
    globalTransform(doc,
            dict(transform=f'rotate({degrees} {px} {py})'))

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
