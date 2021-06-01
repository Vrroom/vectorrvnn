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
    svgs = ['line.svg', 'circle.svg', 'path.svg']
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    for f in svgs : 
        doc = svg.Document(f)
        nPaths = len(doc.paths()) 
        for i in range(nPaths) : 
            for j in range(i + 1, nPaths) : 
                assert(strokeDistance(doc, i, j) < 1e-4)
                assert(fillDistance(doc, i, j) < 1e-4)
                assert(strokeWidthDifference(doc, i, j) < 1e-4)

def test_endpoint () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = ['endpoint_1.svg', 'endpoint_2.svg', 'endpoint_3.svg']
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    docs = [svg.Document(f) for f in svgs]
    assert(endpointDistance(docs[0], 0, 1) < 0.5)
    assert(endpointDistance(docs[1], 0, 1) > 0.7)
    assert(endpointDistance(docs[2], 0, 1) > 0.7)

def test_parallel () :
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = ['parallel_1.svg', 'parallel_2.svg', 
            'parallel_3.svg', 'parallel_4.svg']
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    docs = [svg.Document(f) for f in svgs]
    assert(parallelismDistance(docs[0], 0, 1) < 0.1)
    assert(parallelismDistance(docs[1], 0, 1) < 0.1)
    assert(parallelismDistance(docs[2], 0, 1) < 0.1)
    assert(parallelismDistance(docs[3], 0, 1) > 0.3)

def test_isometry () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    svgs = ['isometry_1.svg', 'isometry_2.svg', 'isometry_3.svg']
    svgs = [osp.join(chdir, 'data', s) for s in svgs]
    docs = [svg.Document(f) for f in svgs]
    assert(isometricDistance(docs[0], 0, 1) < 0.01)
    assert(isometricDistance(docs[1], 0, 1) < 0.01)
    assert(isometricDistance(docs[2], 0, 1) > 0.1)

test_isometry()
