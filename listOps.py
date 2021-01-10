import itertools
from itertools import combinations, chain

def subsets (lst, k) : 
    combs = map(lambda i: combinations(lst, i), range(2, k + 1))
    return itertools.chain(*combs)

def avg (lst) : 
    if len(lst) == 0 : 
        return 0
    return sum(lst) / len(lst)

def hasDuplicates (lst) :
    """
    Check whether a list has duplicates.

    Parameters
    ----------
    lst : list
    """
    return len(set(lst)) != len(lst)

def removeIndices (lst, indices) :
    """
    In place removal of items at given indices. 
    Obtained from : 

    https://stackoverflow.com/questions/497426/deleting-multiple-elements-from-a-list

    Parameters
    ----------
    lst : list
        List from which indices are to be
        removed.
    indices : list
        List of indices. No error checking is 
        done.
    """
    for i in sorted(indices, reverse=True):
        del lst[i] 


def argmax(lst) :
    """
    Compute the argmax of a list.

    Parameters
    ----------
    lst : list
    """
    return next(filter(lambda x : max(lst) == lst[x], range(len(lst))))

def argmin(lst) :
    """
    Compute the argmin of a list.

    Parameters
    ----------
    lst : list
    """
    return next(filter(lambda x : min(lst) == lst[x], range(len(lst))))

def isDisjoint(a, b) : 
    if isinstance(a, int) : 
        a = [a]
    if isinstance(b, int) : 
        b = [b]
    return len(set(a).intersection(set(b))) == 0

def pairwiseDisjoint (setList) :
    """
    Checks whether the sets in the
    list are pairwise disjoint.

    Parameters
    ----------
    setList : list
    """
    for s1, s2 in itertools.combinations(setList, 2) :
        if not s1.isdisjoint(s2) : 
            return False
    return True

def asTuple (a) :
    return a if isinstance(a, tuple) else (a, )
