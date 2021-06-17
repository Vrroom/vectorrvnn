from vectorrvnn.baselines import *
from vectorrvnn.trainutils import Options
from vectorrvnn.utils import *
from vectorrvnn.data import *
import matplotlib.pyplot as plt

def test_svgdata_and_sampler () :
    chdir = osp.split(osp.abspath(__file__))[0]
    opts = Options().parse(testing=[
        '--dataroot',
        osp.join(
            chdir,
            '../../ManuallyAnnotatedDataset_v2'
        ),
        '--name', 
        'test',
        '--n_epochs',
        '1',
        '--batch_size',
        '64',
        '--raster_size',
        '128',
        '--train_epoch_length',
        '256',
        '--val_epoch_length',
        '256'
    ])
    outdir = osp.join(chdir, 'out') 
    files = listdir(osp.join(chdir, 'data'))
    docs = [svg.Document(f) for f in files]
    trees = pmap(suggero, docs)
    # confirm that data points are constructed properly
    datapts = [SVGData(svgFile=f, tree=t) for f, t in zip(files, trees)]
    # confirm that sampling happens properly.
    sampler = AllSampler(datapts, 
            length=opts.train_epoch_length)
    for _ in sampler : 
        pass 
    sampler.reset()
    # confirm that dataloader works properly
    dataloader = TripletDataLoader(opts=opts, sampler=sampler)
    for batch in dataloader : 
        break
    assert(True)

