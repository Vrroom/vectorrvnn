from vectorrvnn.trainutils import Options
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.interfaces import buildData
import random
import matplotlib.image as image

def test_svgdata() :
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot', osp.join(chdir, '../../data/Toy'),
        '--name', 'test',
        '--n_epochs', '1',
        '--modelcls', 'BBoxNet',
    ])
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader, _ = data
    # confirm that dataloader works properly
    for _ in range(opts.n_epochs):  
        for batch in tqdm(trainDataLoader) : 
            pass
    assert(True)

def test_union () :
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot', osp.join(chdir, '../../data/Toy'),
        '--name', 'test',
        '--n_epochs', '1',
        '--modelcls', 'BBoxNet',
        '--train_epoch_length', '256',
        '--val_epoch_length', '256'
    ])
    data = buildData(opts)
    trainData, _, _, _, _ = data
    for i in range(10) : 
        a, b = rng.sample(list(trainData), k=2)
        newPt = a | b 
        assert((not id(newPt) == id(b)) and (not id(newPt) == id(a)))
        assert(newPt.number_of_nodes() == (1 + a.number_of_nodes() + b.number_of_nodes()))
        im = rasterize(newPt.doc, 200, 200)
        fullpath = osp.join('/tmp/', f'union-{i}.png')
        image.imsave(fullpath, im)

def test_union_aug () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot', osp.join(chdir, '../../data/Toy'),
        '--name', 'test',
        '--n_epochs', '1',
        '--modelcls', 'BBoxNet',
        '--train_epoch_length', '256',
        '--val_epoch_length', '256'
    ])
    data = buildData(opts)
    _, valData, _, _, _= data
    aug = GraphicCompose()
    for i in range(4) : 
        graphic = aug(rng.choice(valData), valData)
        im = rasterize(graphic.doc, 200, 200)
        fullpath = osp.join('/tmp/', f'aug-{i}.png')
        image.imsave(fullpath, im)
        figure = treeImageFromGraph(graphic)
        matplotlibFigureSaver(figure,
                osp.join('/tmp/', f'aug-tree-{i}.png'))
    assert(True)

