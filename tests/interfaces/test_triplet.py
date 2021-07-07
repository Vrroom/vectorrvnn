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
        '5',
        '--batch_size',
        '32',
        '--raster_size',
        '128',
        '--train_epoch_length',
        '128',
        '--val_epoch_length',
        '128',
        '--decay_start',
        '0',
        '--samplercls',
        'DiscriminativeSampler',
        '--modelcls',
        'PatternGrouping',
        '--structure_embedding_size',
        '8',
        '--augmentation',
        'compose'
    ])
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader = data
    model = buildModel(opts) 
    interface = TripletInterface(opts, model, trainData, valData)
    trainer = ttools.Trainer(interface)
    addCallbacks(trainer, model, data, opts)
    # Start training
    import pdb
    pdb.set_trace()
    trainer.train(
        trainDataLoader, 
        num_epochs=opts.n_epochs, 
        val_dataloader=valDataLoader
    )
    assert(True)

test_interface()
