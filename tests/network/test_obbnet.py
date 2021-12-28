import torch
from vectorrvnn.utils import * 
from vectorrvnn.network import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from vectorrvnn.interfaces import *
import os
import os.path as osp
from copy import deepcopy

def test_model () : 
    """ test whether things work """
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot', 
        osp.join(chdir, '../../data/Toy'),
        '--name', 'test', 
        '--embedding_size', '64',
        '--encoder_layers', '1',
        '--heads', '1',
        '--hidden_size', '128', '128',
        '--modelcls', 'OBBNet', 
        '--n_epochs', '10',
        '--batch_size', '1'
    ])
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader, _ = data
    model = buildModel(opts) 
    interface = TripletInterface(opts, model, trainData, valData)
    trainer = ttools.Trainer(interface)
    addCallbacks(trainer, model, data, opts)
    for batch in trainDataLoader : 
        break
    f1 = model.embedding(batch['ref'])
    for i in range(opts.max_len) :
        ref = deepcopy(batch['ref'])
        masked = ref['mask'][0, i]
        ref['obb'][:, i, :] = 1.0
        f2 = model.embedding(ref)
        print(torch.linalg.norm(f2 - f1))
    assert True

test_model()
