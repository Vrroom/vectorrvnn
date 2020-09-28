import svgpathtools as svg
import itertools 
import more_itertools
import networkx as nx
import xml.etree.ElementTree as ET
from treeOps import removeOneOutDegreeNodesFromTree

def graphFromSvg (svgFile, graphFunction) : 
    """
    Produce a relationship graph for 
    the svg file

    Parameters
    ----------
    svgFile : str
        path to svg file.
    graphFunction : lambda 
        path relation computer.
    """
    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    G = graphFunction(paths, vbox=doc.get_viewbox())
    return G

def relationshipGraph (svgFile, relFunctions) :
    """
    Take the svg, find graphs
    for all relationship types
    and combine them into one single
    graph for clustering.

    Parameters
    ----------
    svgFile : str
        path to svg file.
    relFunctions : list
        List of graph computing functions
    """
    graphs = map(lambda x : graphFromSvg(svgFile, x), relFunctions)
    graph = None
    for g in graphs :
        if graph is None:
            graph = g
        else :
            graph = nx.compose(graph, g)
    return graph

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

    childTags = [
        '{http://www.w3.org/2000/svg}rect',
        '{http://www.w3.org/2000/svg}circle',
        '{http://www.w3.org/2000/svg}ellipse',
        '{http://www.w3.org/2000/svg}line', 
        '{http://www.w3.org/2000/svg}polyline',
        '{http://www.w3.org/2000/svg}polygon',
        '{http://www.w3.org/2000/svg}path',
    ]
    groupTag = '{http://www.w3.org/2000/svg}g'
    doc = svg.Document(svgFile)
    paths = doc.flatten_all_paths()
    zIndexMap = dict([(p.zIndex, i) for i, p in enumerate(paths)])
    tree = doc.tree
    root = tree.getroot()
    allNodes = list(root.iter())
    T = nx.DiGraph()
    r = 0
    buildTreeGraph (root)
    T = removeOneOutDegreeNodesFromTree(T)
    return T

def tree2Document (document, tree, attribs) :
    """
    Add a tree of parents to svg document.

    Parameters
    ----------
    document : svg.Document
        SVG document to which the tree of paths 
        has to be added.
    tree : nx.DiGraph
        Path hierarchy.
    attribs : dict
        Dictionary of path attributes. Same attributes
        are applied to all paths.
    """
    def addToDocument (T, r, children) :
        if T.out_degree(r) == 0 : 
            document.add_path(T.nodes[r]['path'], attribs=attribs, group=parentGroup[r])
        else :
            newGroup = document.add_group(parent=parentGroup[r])
            for child in children: 
                parentGroup[child] = newGroup

    root = findRoot(tree)
    parentGroup = {root: None}
    treeApplyRootFirst(tree, root, addToDocument)
    return document
