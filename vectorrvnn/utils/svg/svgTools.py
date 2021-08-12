from functools import lru_cache, wraps
import re
import string
from matplotlib import colors
import xml.etree.ElementTree as ET
from copy import deepcopy
import networkx as nx
import itertools, more_itertools
import svgpathtools as svg
from vectorrvnn.utils.graph import *
from vectorrvnn.utils.boxes import *

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

PATH_TAGS = GRAPHIC_TAGS[:-1]

class Del:
    def __init__(self, keep=string.digits + '.-e'):
        self.comp = dict((ord(c),c) for c in keep)
    def __getitem__(self, k):
        return self.comp.get(k)
DD = Del()

def immutable_doc (fn) :
    @wraps(fn)
    def wrapper (doc, *args, **kwargs) :
        doc_ = deepcopy(doc)
        return fn(doc_, *args, **kwargs)
    return wrapper

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

    childTags = PATH_TAGS
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
    n = len(leaves(T))
    leafLabels = dict(map(reversed, enumerate(leaves(T))))
    nonLeafLabels = dict([(_, n + i) for i, _ in enumerate(nonLeaves(T))])
    T = nx.relabel_nodes(T, mapping={**leafLabels, **nonLeafLabels})
    return T

@lru_cache(maxsize=1024)
def cachedPaths (doc) : 
    return doc.paths()

def unparseCSS(cssDict) :  
    items = [f'{k}:{v}' for k, v in cssDict.items()]
    return ';'.join(items)

def parseCSS(cssString) : 
    return dict(map(
        lambda x : x.split(':'), 
        cssString.split(';')
    ))

def parseDashArray(da) : 
    return list(map(float, filter(
        lambda x : len(x) > 0, 
        re.split(',| ', da))
    ))

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

def xmlAttributeSet (element, attr, value) : 
    if value is None: return 
    element.attrib[attr] = value
    if 'style' in element.attrib : 
        style = element.attrib['style']
        parsed = parseCSS(style)
        if attr in parsed : 
            parsed[attr] = value
            element.attrib['style'] = unparseCSS(parsed)

def xmlAttributeGet (element, attr, default) : 
    if attr in element.attrib : 
        return element.attrib[attr]
    elif 'style' in element.attrib : 
        style = element.attrib['style']
        parsed = parseCSS(style)
        if attr in parsed : 
            return parsed[attr]
    return default

def pathAttributeSet (path, attr, value) : 
    xmlAttributeSet(path.element, attr, value)

def pathAttributeGet (path, attr, default) : 
    return xmlAttributeGet(path.element, attr, default)
    
def pathColor (path, attr) : 
    """ attr can be one of stroke/fill """
    return parseColor(pathAttributeGet(path, attr, DEFAULT_COLOR))

def pathStrokeWidth (path) :
    pw = pathAttributeGet(path, 'stroke-width', DEFAULT_STROKEWIDTH)
    return float(pw.translate(DD))

def pathStrokeLineCap (path): 
    return pathAttributeGet(path, 'stroke-linecap', DEFAULT_LINECAP)

def pathStrokeLineJoin (path): 
    return pathAttributeGet(path, 'stroke-linejoin', DEFAULT_LINEJOIN)

def pathStrokeDashArray(path) : 
    """ 
    Instead of the whole dasharray string, just 
    say whether there is this attribute or not
    """
    da = pathAttributeGet(path, 'stroke-dasharray', None)
    return da if da != 'none' else None

def fixOrigin (doc) : 
    x, y, w, h = getDocBBox(doc).tolist()
    globalTransform(doc, 
            dict(transform=f'translate({-x:.3f} {-y:.3f})'))
    setDocBBox(doc, BBox(0, 0, w, h, w, h))

def scaleToFit (doc, w, h) : 
    box = getDocBBox(doc)
    ox, oy, ow, oh = box.tolist()
    sx = w / ow
    sy = h / oh
    globalTransform(doc, 
            dict(transform=f'scale({sx:.3f} {sy:.3f})'))
    setDocBBox(doc, box.scaled(sx, sy))

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
    # rootCopy = deepcopy(doc.tree.getroot())
    root = doc.tree.getroot()
    # Don't add <defs ... /> to new group element.
    children = [e for e in root if e.tag in GRAPHIC_TAGS]
    groupElement.extend(children)
    # Remove graphic elements.
    removeGraphicChildren(root)
    root.append(groupElement)
    doc.tree._setroot(root)
    doc.updateParentMap()

@immutable_doc
def crop (doc, box) :
    setDocBBox(doc, box.normalized())
    return doc

def iterfilter (root, tagFilter) : 
    if not tagFilter(root.tag) : 
        yield root 
        for child in root : 
            yield from iterfilter(child, tagFilter)

@immutable_doc
def subsetSvg(doc, lst) :
    root = doc.tree.getroot()
    paths = list(filter(
        lambda e : e.tag in PATH_TAGS, 
        iterfilter(root, lambda tag : tag == '{http://www.w3.org/2000/svg}defs')
    ))
    n = len(paths)
    unwanted = list(set(range(n)) - set(lst))
    unwantedElts = [paths[i] for i in unwanted]
    doc.updateParentMap()
    for elt in unwantedElts : 
        doc.parent_map[elt].remove(elt)
    newDocument = svg.Document(None)
    newDocument.fromString(ET.tostring(root, encoding='unicode'))
    return newDocument

@immutable_doc
def withoutDegeneratePaths (doc) : 
    root = doc.tree.getroot()
    paths = doc.paths()
    degenZIndices = [p.zIndex for p in paths 
            if pathBBox(p.path).isDegenerate()]
    allElts = list(root.iter())
    unwantedElts = [allElts[i] for i in degenZIndices]
    doc.updateParentMap()
    for elt in unwantedElts :
        doc.parent_map[elt].remove(elt)
    newDocument = svg.Document(None)
    newDocument.fromString(ET.tostring(root, encoding='unicode'))
    return newDocument

@immutable_doc
def docUnion (doc, that) :
    # add the canvases
    setDocBBox(doc, getDocBBox(doc) | getDocBBox(that))
    # add the paths
    thatRoot = deepcopy(that.tree.getroot())
    thatChildren = list(filter(
        lambda e : e.tag in GRAPHIC_TAGS,
        thatRoot
    ))
    root = doc.tree.getroot()
    root.extend(thatChildren)
    return doc
