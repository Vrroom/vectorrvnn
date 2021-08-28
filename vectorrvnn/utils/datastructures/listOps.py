import itertools
from itertools import combinations, chain

def subsets (lst, k) : 
    """ return size k subsets of list """ 
    combs = map(lambda i: combinations(lst, i), range(2, k + 1))
    return itertools.chain(*combs)

def avg (lst) : 
    """ 
    use this instead of np.mean if you are worried
    about empty lists. if list is actually an iterator,
    this function will consume the iterator.
    """
    lst = list(lst)
    if len(lst) == 0 : 
        return 0
    return sum(lst) / len(lst)

def hasDuplicates (lst) :
    """
    Check whether a list has duplicates.
    """
    return len(set(lst)) != len(lst)

def removeIndices (lst, indices) :
    """
    In place removal of items at given indices. Obtained from : 

        https://stackoverflow.com/questions/497426/deleting-multiple-elements-from-a-list
    """
    for i in sorted(indices, reverse=True):
        del lst[i] 

def argmax(lst) :
    """
    Compute the argmax of an iterator
    """
    lst = list(lst)
    m = max(lst)
    return next(filter(lambda x : m == lst[x], range(len(lst))))

def argmin(lst) :
    """
    Compute the argmin of an iterator
    """
    lst = list(lst)
    m = min(lst)
    return next(filter(lambda x : m == lst[x], range(len(lst))))

def isDisjoint(a, b) : 
    """ are two lists disjoint """
    return len(set(a).intersection(set(b))) == 0

def pairwiseDisjoint (setList) :
    """
    Checks whether the sets in the list are pairwise disjoint.
    """
    for s1, s2 in itertools.combinations(setList, 2) :
        if not s1.isdisjoint(s2) : 
            return False
    return True
