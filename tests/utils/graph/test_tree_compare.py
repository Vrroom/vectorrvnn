from vectorrvnn.utils import *

def test_levenshtein () :
    skipSpace = lambda x : 0 if x == ' ' else 1
    match = lambda a, b : 0 if a == b else 1
    assert(levenshteinDistance('sumit chaturvedi', 
        'su mi t  chat ur vedi', match, skipSpace) == 0)
    assert(levenshteinDistance('suummit', 'sumit',
        match, skipSpace) == 2)

def test_cted () : 
    a = nx.DiGraph()
    a.add_edges_from([(0, 1), (0, 2), (2, 3), (0, 4), (4, 5)])
    for n in a.nodes:
        a.nodes[n]['pathSet'] = leavesInSubtree(a, n)
    opt, matching = cted(a, a, True)
    assert(opt == 0)
