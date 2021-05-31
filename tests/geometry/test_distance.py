from vectorrvnn.geometry.distance import *
from vectorrvnn.utils import *
import os
import os.path as osp
import svgpathtools as svg

def fpAssert (v1, v2) : 
    assert(abs(v1 - v2) < 1e-4)

def test_proximity () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    circleDoc = svg.Document(osp.join(chdir, 'data', 'circle.svg'))
    lp = localProximity(circleDoc, 0, 1)
    gp = globalProximity(circleDoc, 0, 1)
    fpAssert(lp, 40 * np.sqrt(2))
    fpAssert(gp, 40 * np.sqrt(2))

def test_pathattrs () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    for f in listdir(osp.join(chdir, 'data')) : 
        doc = svg.Document(f)
        nPaths = len(doc.paths()) 
        for i in range(nPaths) : 
            for j in range(i + 1, nPaths) : 
                assert(strokeDistance(doc, i, j) < 1e-4)
                assert(fillDistance(doc, i, j) < 1e-4)
                assert(strokeWidthDifference(doc, i, j) < 1e-4)

test_pathattrs()
