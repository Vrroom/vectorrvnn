from vectorrvnn.utils import *

def test_levenshtein () :
    skipSpace = lambda x : 0 if x == ' ' else 1
    match = lambda a, b : 0 if a == b else 1
    assert(levenshteinDistance('sumit chaturvedi', 
        'su mi t  chat ur vedi', match, skipSpace) == 0)
    assert(levenshteinDistance('suummit', 'sumit',
        match, skipSpace) == 2)
