from vectorrvnn.geometry.descriptor import *
from vectorrvnn.geometry.distance import *
from vectorrvnn.utils import *
import os
import os.path as osp
import svgpathtools as svg
from scipy.stats import wasserstein_distance

def test_d2 () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    circleDoc = svg.Document(osp.join(chdir, 'data', 'circle.svg'))
    lineDoc = svg.Document(osp.join(chdir, 'data', 'line.svg'))

    circle_d2_1 = d2(circleDoc, 0, nSamples=1000)
    circle_d2_2 = d2(circleDoc, 1, nSamples=1000)

    line_d2_1 = d2(lineDoc, 0, nSamples=1000)
    line_d2_2 = d2(lineDoc, 1, nSamples=1000)

    assert(wasserstein_distance(circle_d2_1, circle_d2_2) < 1.5e-2)
    assert(wasserstein_distance(line_d2_1, line_d2_2) < 1.5e-2)
    assert(wasserstein_distance(circle_d2_1, line_d2_1) > 2e-2)
    assert(wasserstein_distance(circle_d2_2, line_d2_2) > 2e-2)

def test_fd () :
    chdir = osp.split(osp.abspath(__file__))[0]
    pathDoc = svg.Document(osp.join(chdir, 'data', 'path.svg'))
    circleDoc = svg.Document(osp.join(chdir, 'data', 'circle.svg'))

    fd1 = fd(pathDoc, 0)
    fd2 = fd(pathDoc, 1)
    fd3 = fd(pathDoc, 2)
    fd4 = fd(circleDoc, 0)

    assert(np.linalg.norm(fd1 - fd2) < 1e-3)
    assert(np.linalg.norm(fd2 - fd3) < 1e-3)
    assert(np.linalg.norm(fd3 - fd4) > 1)
    # for a circle only the DC component will be there
    assert(np.abs(fd4[1:]).sum() < 1e-1)
    assert(fourierDescriptorDistance(pathDoc, 0, 1) < 1e-4)

def test_bb () :
    chdir = osp.split(osp.abspath(__file__))[0]
    circleDoc = svg.Document(osp.join(chdir, 'data', 'circle.svg'))
    # first circle is well within the bounding box.
    b1 = bb(circleDoc, 0)
    b2 = relbb(circleDoc, 0)
    assert(abs(b1[2] - b1[3]) < 1e-4)
    assert(all([0 <= b <= 1 for b in b2]))

