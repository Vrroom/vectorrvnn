from vectorrvnn.baselines import *
from vectorrvnn.trainutils import *
from vectorrvnn.utils import *
from vectorrvnn.data import *
from tqdm import tqdm
import os
import os.path as osp
import networkx as nx
import svgpathtools as svg
from more_itertools import unzip, flatten
import logging

logging.basicConfig(filename='.log', level=logging.INFO)

def test_suggero () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    data = [SVGData(f) for f in files]
    trees = list(map(suggero, data))
    for t, f in zip(trees, files) : 
        fname = getBaseName(f)
        figure = treeImageFromGraph(t)
        matplotlibFigureSaver(figure,
                osp.join('/tmp/', 'suggero-' + fname))
    assert(True)

def test_distance_from_self_is_zero () :
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    data = [SVGData(f) for f in files]
    trees = pmap(suggero, data)
    for t in trees : 
        assert(ted(t, t) < 1e-4)

def test_autogroup () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    data = [SVGData(f) for f in files]
    trees = list(map(autogroup, data))
    for t, f in zip(trees, files) : 
        fname = getBaseName(f)
        figure = treeImageFromGraph(t)
        matplotlibFigureSaver(figure,
                osp.join('/tmp/', 'autogroup-' + fname))
    assert(True)
