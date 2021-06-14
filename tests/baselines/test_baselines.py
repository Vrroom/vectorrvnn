from vectorrvnn.baselines import *
from vectorrvnn.utils import *
import os
import os.path as osp
import networkx as nx
import svgpathtools as svg
from more_itertools import unzip
import logging

logging.basicConfig(filename='.log', level=logging.INFO)

def test_suggero () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    docs = [svg.Document(f) for f in files]
    trees = list(map(suggero, docs))
    for t, d, f in zip(trees, docs, files) : 
        fname = getBaseName(f)
        t.doc = d
        figure = treeImageFromGraph(t)
        matplotlibFigureSaver(figure,
                osp.join(chdir, 'out', 'suggero-' + fname))
    assert(True)

def test_distance_from_self_is_zero () :
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    docs = [svg.Document(f) for f in files]
    trees = pmap(suggero, docs)
    for t in trees : 
        assert(ted(t, t) < 1e-4)

def test_autogroup () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    docs = [svg.Document(f) for f in files]
    trees = list(map(autogroup, docs))
    for t, d, f in zip(trees, docs, files) : 
        fname = getBaseName(f)
        t.doc = d
        figure = treeImageFromGraph(t)
        matplotlibFigureSaver(figure,
                osp.join(chdir, 'out', 'autogroup-' + fname))
    assert(True)

def logScores (ts1, ts2) : 
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    tedscores = list(map(scoreFn, ts1, ts2))
    fmi1scores = list(map(partial(fmi, level=1), ts1, ts2))
    fmi2scores = list(map(partial(fmi, level=2), ts1, ts2))
    fmi3scores = list(map(partial(fmi, level=3), ts1, ts2))
    logging.info(f'T.E.D.    = {avg(tedscores)}')
    logging.info(f'F.M.I.(1) = {avg(fmi1scores)}')
    logging.info(f'F.M.I.(2) = {avg(fmi2scores)}')
    logging.info(f'F.M.I.(3) = {avg(fmi3scores)}')

def test_scores () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    dataDir = osp.join(chdir, '../../ManuallyAnnotatedDataset_v2/Val')
    files = listdir(dataDir)[:5]
    svgFiles = [osp.join(f, osp.split(f)[1] + '.svg') for f in files]
    treeFiles = [osp.join(f, osp.split(f)[1] + '.pkl') for f in files]
    docs = [svg.Document(f) for f in svgFiles]
    small = [(t, s, d) for t, s, d in zip(treeFiles, svgFiles, docs) 
            if len(d.paths()) < 50]
    treeFiles, svgFiles, docs = list(map(list, unzip(small)))
    gtTrees = list(map(nx.read_gpickle, treeFiles))
    autogroupTrees = list(map(autogroup, docs))
    suggeroTrees = list(map(suggero, docs))
    logging.info('Autogroup')
    logScores(autogroupTrees, gtTrees)
    logging.info('Suggero')
    logScores(suggeroTrees, gtTrees)
    assert(True)

test_suggero()
