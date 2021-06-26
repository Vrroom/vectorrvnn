from vectorrvnn.utils import *
from itertools import product
import os
import os.path as osp
import svgpathtools as svg

def test_boxes () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    docs = list(map(svg.Document, files))
    for doc in docs : 
        docbox = getDocBBox(doc)
        assert(docbox + docbox == docbox)
        assert(abs((docbox / docbox).area() - 1) < 1e-4)
        assert(abs((docbox / docbox).center() - complex(0.5, 0.5)) < 1e-4)
        paths = cachedPaths(doc)
        for path in paths : 
            pathbox = pathBBox(path.path)
            assert(pathbox + pathbox == pathbox)

        for pi, pj in product(paths, paths):
            b1 = pathBBox(pi.path)
            b2 = pathBBox(pj.path) 
            assert(implies(b1 in b2, not b2 in b1))
            assert(implies(b1 in b2, not b2 in b1))

