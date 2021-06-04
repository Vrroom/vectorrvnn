from vectorrvnn.baselines import *
from vectorrvnn.utils import *
from vectorrvnn.data.SVGData import SVGData

def test_svgdata () :
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    docs = [svg.Document(f) for f in files]
    trees = pmap(suggero, docs)
    datapts = [SVGData(svgFile=f, tree=t) for f, t in zip(files, trees)]
    assert(True)
