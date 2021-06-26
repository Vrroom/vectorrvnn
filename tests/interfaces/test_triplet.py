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
        '1',
        '--batch_size',
        '2',
        '--raster_size',
        '128',
        '--train_epoch_length',
        '256',
        '--val_epoch_length',
        '256',
        '--decay_start',
        '0',
        '--samplercls',
        'DiscriminativeSampler',
        '--modelcls',
        'PatternGroupingV2',
        '--structure_embedding_size',
        '8',
        '--augmentation',
        'simple'
    ])
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader = data
    model = buildModel(opts) 
    interface = TripletInterface(opts, model, trainData, valData)
    trainer = ttools.Trainer(interface)
    addCallbacks(trainer, model, data, opts)
    # Start training
    import cProfile
    pro = cProfile.Profile()
    pro.enable()
    trainer.train(
        trainDataLoader, 
        num_epochs=opts.n_epochs, 
        val_dataloader=valDataLoader
    )
    pro.disable()
    pro.print_stats('cumtime')
    assert(True)

test_interface()
