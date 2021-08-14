from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from vectorrvnn.network import *
from vectorrvnn.interfaces import buildData, buildModel, TripletInterface, addCallbacks
import ttools

def test_interface() : 
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot',
        osp.join(chdir, '../../data/Toy'),
        '--name', 
        'test',
        '--n_epochs',
        '3',
        '--batch_size',
        '4',
        '--K',
        '2',
        '--train_epoch_length',
        '128',
        '--val_epoch_length',
        '128',
        '--decay_start',
        '0',
        '--embedding_size',
        '32',
        '--samplercls',
        'DiscriminativeSampler',
        '--modelcls',
        'OneBranch',
        '--augmentation',
        'multiaug',
        '--frequency',
        '1',
        '--loss',
        'hardCosineSimilarity',
        '--freeze_layers',
        ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']
    ])
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader = data
    model = buildModel(opts) 
    interface = TripletInterface(opts, model, trainData, valData)
    trainer = ttools.Trainer(interface)
    addCallbacks(trainer, model, data, opts)
    # Start training
    trainer.train(
        trainDataLoader, 
        num_epochs=opts.n_epochs, 
        val_dataloader=valDataLoader
    )
    assert(True)

