from vectorrvnn.baselines import *
from vectorrvnn.utils import *
from vectorrvnn.data import *
import matplotlib.pyplot as plt

def test_svgdata_and_sampler () :
    chdir = osp.split(osp.abspath(__file__))[0]
    outdir = osp.join(chdir, 'out') 
    files = listdir(osp.join(chdir, 'data'))
    docs = [svg.Document(f) for f in files]
    trees = pmap(suggero, docs)
    # confirm that data points are constructed properly
    datapts = [SVGData(svgFile=f, tree=t) for f, t in zip(files, trees)]
    # confirm that sampling happens properly.
    sampler = TripletSampler(datapts, length=8)
    for _ in sampler : 
        pass 
    sampler.reset()
    # confirm that dataloader works properly
    dataloader = TripletDataLoader(batch_size=4, sampler=sampler)
    for batch in dataloader : 
        break
    # visualize the triplets in batch as images
    refWhole = batch['refWhole'].permute((0, 2, 3, 1)).numpy()
    plusWhole = batch['plusWhole'].permute((0, 2, 3, 1)).numpy()
    minusWhole = batch['minusWhole'].permute((0, 2, 3, 1)).numpy()
    big = np.concatenate((refWhole, plusWhole, minusWhole), axis=2)
    big = big.reshape(-1, *big.shape[2:])
    big = (big - big.min()) / (big.max() - big.min())
    plt.imshow(big)
    plt.savefig(osp.join(outdir, 'triplet.png'))
    assert(True)

