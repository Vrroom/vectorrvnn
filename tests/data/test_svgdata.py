from vectorrvnn.trainutils import Options
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.interfaces import buildData
import random
import matplotlib.image as image

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
        '32',
        '--train_epoch_length',
        '256',
        '--val_epoch_length',
        '256',
        '--decay_start',
        '0',
        '--samplercls',
        'DiscriminativeSampler',
        '--modelcls',
        'OneBranch',
        '--augmentation',
        'multiaug'
    ])
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader = data
    # confirm that dataloader works properly
    for batch in tqdm(trainDataLoader) : 
        pass
    assert(True)

def test_union () :
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot',
        osp.join(chdir, '../../data/Toy'),
        '--name', 
        'test',
        '--n_epochs',
        '2',
        '--batch_size',
        '32',
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
    trainData, _, _, _ = data
    for i in range(10) : 
        a, b = random.sample(list(trainData), k=2)
        newPt = a | b 
        assert((not id(newPt) == id(b)) and (not id(newPt) == id(a)))
        assert(newPt.number_of_nodes() == (1 + a.number_of_nodes() + b.number_of_nodes()))
        im = rasterize(newPt.doc, 200, 200)
        fullpath = osp.join(chdir, 'out', f'union-{i}.png')
        image.imsave(fullpath, im)

def test_union_aug () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot', osp.join(chdir, '../../data/Toy'),
        '--embedding_size', '32',
        '--samplercls', 'DiscriminativeSampler',
        '--phase', 'test'
    ])
    data = buildData(opts)
    _, valData, _, _ = data
    aug = GraphicCompose()
    for i in range(4) : 
        graphic = aug(trng.choice(valData), valData)
        im = rasterize(graphic.doc, 200, 200)
        fullpath = osp.join(chdir, 'out', f'aug-{i}.png')
        image.imsave(fullpath, im)
        figure = treeImageFromGraph(graphic)
        matplotlibFigureSaver(figure,
                osp.join(chdir, 'out', f'aug-tree-{i}.png'))
    assert(True)

