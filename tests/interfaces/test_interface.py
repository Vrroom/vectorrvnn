from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from vectorrvnn.network import *
from vectorrvnn.interfaces import *
import ttools

def test_interface() : 
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
        '--n_epochs', '2',
        '--base_size', '1',
        '--batch_size', '32',
        '--train_epoch_length', '128',
        '--val_epoch_length', '128',
        '--dataloadercls', 'ContrastiveDataLoader',
        '--samplercls', 'ContrastiveSampler',
        '--loss', 'supCon'
    ])
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader, _ = data
    model = buildModel(opts) 
    interface = Interface(opts, model, trainData, valData)
    trainer = ttools.Trainer(interface)
    # Start training
    trainer.train(
        trainDataLoader, 
        num_epochs=opts.n_epochs, 
        val_dataloader=valDataLoader
    )
    assert(True)

