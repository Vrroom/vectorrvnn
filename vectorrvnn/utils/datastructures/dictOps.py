from more_itertools import flatten
from functools import partial

def aggregateDict (listOfDicts, reducer, keys=None) : 
    """ 
    Very handy function to combine a list of dicts
    into a dict with the reducer applied by key.
    """
    if keys is None : 
        keys = list(set(flatten(map(deepKeys, listOfDicts))))
    aggregator = lambda key : reducer(
        list(map(
            partial(deepGet, deepKey=key), 
            listOfDicts
        ))
    )
    return deepDict(zip(keys, map(aggregator, keys)))

def dictmap (f, d) : 
    new = dict()
    for k, v in d.items() : 
        new[k] = f(k, v)
    return new

def deepKeys (dictionary) : 
    """ iterate over keys of dict of dicts """
    stack = [((), dictionary)]
    while len(stack) > 0 : 
        prevKeys, dictionary = stack.pop()
        for k, v in dictionary.items() : 
            if isinstance(v, dict) :
                stack.append(((*prevKeys, k), v))
            else : 
                yield (*prevKeys, k)

def deepGet (dictionary, deepKey) : 
    """ get key in a dict of dicts """
    v = dictionary[deepKey[0]] 
    if isinstance(v, dict) : 
        return deepGet(v, deepKey[1:])
    else : 
        return v

def deepDict (pairs) : 
    """ 
    Create a deep dict a.k.a a dict of dicts
    where a key may be tuple
    """
    d = dict()
    for k, v in pairs : 
        d_ = d
        for k_ in k[:-1] : 
            if k_ not in d_ : 
                d_[k_] = dict()
            d_ = d_[k_]
        d_[k[-1]] = v
    return d

