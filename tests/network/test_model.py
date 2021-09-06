import torch
from vectorrvnn.utils import * 
from vectorrvnn.network import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from vectorrvnn.interfaces import *
import os
import os.path as osp
import logging
from tqdm import tqdm

def test_model () : 
    """ test whether things work """
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot', 
        osp.join(chdir, '../../data/Toy'),
        '--name', 
        'test', 
        '--embedding_size',
        '32',
        '--modelcls', 
        'OneBranch',
    ])
    test(opts)
    assert True

