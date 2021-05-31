import more_itertools

def aggregateDict (listOfDicts, reducer) : 
    """ 
    Very handy function to combine a list of dicts
    into a dict with the reducer applied by key.
    """
    keys = set(more_itertools.flatten(map(lambda x : x.keys(), listOfDicts)))
    aggregator = lambda key : reducer(list(map(lambda x : x[key], listOfDicts)))
    return dict(zip(keys, map(aggregator, keys)))
