import torch
from vectorrvnn.utils import * 
from vectorrvnn.network import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from vectorrvnn.interfaces import buildData, buildModel, TripletInterface, addCallbacks
import os
import os.path as osp
import logging
from tqdm import tqdm

logging.basicConfig(filename='.log', level=logging.INFO)

def logScores (ts1, ts2) : 
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    tedscores = list(map(scoreFn, tqdm(ts1), ts2))
    fmi1scores = list(map(partial(fmi, level=1), ts1, ts2))
    fmi2scores = list(map(partial(fmi, level=2), ts1, ts2))
    fmi3scores = list(map(partial(fmi, level=3), ts1, ts2))
    logging.info(f'T.E.D.    = {avg(tedscores)}')
    logging.info(f'F.M.I.(1) = {avg(fmi1scores)}')
    logging.info(f'F.M.I.(2) = {avg(fmi2scores)}')
    logging.info(f'F.M.I.(3) = {avg(fmi3scores)}')

def test_model () : 
    """ test whether things work """
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot', osp.join(chdir, '../../ManuallyAnnotatedDataset_v2'),
        '--checkpoints_dir', osp.join(chdir, '../../results'),
        '--name', 'test_model',
    ])
    data = TripletDataset(osp.join(opts.dataroot, 'Test'))
    data = [t for t in data if t.nPaths < 50][:10]
    model = buildModel(opts)
    data = list(map(forest2tree, data))
    out = list(map(model.greedyTree, tqdm(data)))
    logScores(data, out)
    assert(True)
