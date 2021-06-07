import os
import os.path as osp
import svgpathtools as svg
import networkx as nx
from vectorrvnn.utils import * 

def test_svg2tree (): 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgFile = osp.join(
        chdir, 
        'data', 
        '20180-Vector-image-of-house-with-chimney-on-green-grass.svg'
    )
    doc = svg.Document(svgFile)
    nPaths = len(doc.paths())
    t = getTreeStructureFromSVG(svgFile) 
    assert (nx.is_tree(t))
    assert (len(leaves(t)) == nPaths) 
    assert (all([t.out_degree(n) > 1 for n in nonLeaves(t)]))
    # visualize the tree structure
    t.doc = doc
    figure = treeImageFromGraph(t)
    matplotlibFigureSaver(figure, 
            osp.join(chdir, 'out', 'svg2tree'))

