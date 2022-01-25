from vectorrvnn.utils import *

def test_binarize() :
    a = nx.DiGraph()
    a.add_edges_from([(0, 1), (0, 2), (0, 5), (5, 4), (5, 3)])
    b = randomBinaryTree(a)
    assert(sorted(leaves(a)) == sorted(leaves(b)))
    assert(all([b.out_degree(_) == 2 for _ in nonLeaves(b)]))

