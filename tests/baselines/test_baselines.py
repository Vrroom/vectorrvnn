from vectorrvnn.baselines import *
from vectorrvnn.utils import *
import os
import os.path as osp
import svgpathtools as svg
import cProfile

def test_suggero () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    docs = [svg.Document(f) for f in files]
    trees = pmap(suggero, docs)
    for t, d, f in zip(trees, docs, files) : 
        fname = getBaseName(f)
        t.doc = d
        figure = treeImageFromGraph(t)
        matplotlibFigureSaver(figure,
                osp.join(chdir, 'out', fname))
    assert(True)

def test_distance_from_self_is_zero () :
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    docs = [svg.Document(f) for f in files]
    trees = pmap(suggero, docs)
    for t in trees : 
        assert(ted(t, t) < 1e-4)

