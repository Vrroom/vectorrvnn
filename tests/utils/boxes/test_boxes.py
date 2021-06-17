from vectorrvnn.utils import *
from itertools import product
import os
import os.path as osp
import svgpathtools as svg

def implies(a, b) :
    return not a or b

def test_boxes () : 
    chdir = osp.split(osp.abspath(__file__))[0]
    files = listdir(osp.join(chdir, 'data'))
    docs = list(map(svg.Document, files))
    for doc in docs : 

        docbox = getDocBBox(doc)
        assert(docbox.todim() == docbox.todim())
        assert(docbox.toext() == docbox.toext())
        assert(docbox + docbox == docbox)

        paths = cachedPaths(doc)
        for path in paths : 
            pathbox = pathBBox(path.path)
            assert(pathbox.todim() == pathbox.todim())
            assert(pathbox.toext() == pathbox.toext())
            assert(pathbox + pathbox == pathbox)
            assert(pathbox.isDegenerate() == pathbox.todim().isDegenerate())
            assert(pathbox.isDegenerate() == pathbox.toext().isDegenerate())
            assert(abs(pathbox.todim().center() - pathbox.toext().center()) < 1e-4)
            assert(pathbox.todim().normalized().toext().iou(pathbox.toext().normalized()) > 0.99)
            assert(pathbox.toext().normalized().todim().iou(pathbox.todim().normalized()) > 0.99)

        for pi, pj in product(paths, paths):
            b1 = pathBBox(pi.path)
            b2 = pathBBox(pj.path) 
            assert(implies(b1.toext().contains(b2.toext()), (not b2.toext().contains(b1.toext()))))
            assert(implies(b1.todim().contains(b2.todim()), (not b2.todim().contains(b1.todim()))))
            assert(abs((b1 + b2).center() - (b1.toext() + b2.toext()).center()) < 1e-4)
            assert(abs((b1 + b2).center() - (b1.todim() + b2.todim()).center()) < 1e-4)

