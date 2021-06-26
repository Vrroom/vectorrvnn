from vectorrvnn.trainutils import Options
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.interfaces import *

def test_svgdata() :
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot',
        osp.join(chdir, '../../data/Toy'),
        '--name', 
        'test',
        '--n_epochs',
        '2',
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
        'none'
    ])
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader = data
    # confirm that dataloader works properly
    for batch in tqdm(trainDataLoader) : 
        pass
    assert(True)

