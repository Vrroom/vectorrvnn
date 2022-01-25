import os
import os.path as osp
import svgpathtools as svg
import networkx as nx
from vectorrvnn.utils import * 
from vectorrvnn.data import *

def test_svg2tree (): 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgFile = osp.join(
        chdir, 
        'data', 
        '20180-Vector-image-of-house-with-chimney-on-green-grass.svg'
    )
    t = SVGData(svgFile)
    assert (nx.is_tree(t))
    assert (len(leaves(t)) == t.nPaths) 
    assert (all([t.out_degree(n) > 1 for n in nonLeaves(t)]))
    # visualize the tree structure
    figure = treeImageFromGraph(t)
    matplotlibFigureSaver(figure, 
            osp.join('/tmp/', 'svg2tree'))

