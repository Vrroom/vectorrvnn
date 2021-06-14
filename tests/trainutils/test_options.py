import os
import os.path as osp
from vectorrvnn.trainutils import Options

def test_options () :
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
    assert True
