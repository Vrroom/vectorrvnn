from vectorrvnn.utils import *
from itertools import permutations, combinations, starmap
from functools import partial
from copy import deepcopy 
from .Sampler import *

class DiscriminativeSampler (Sampler):  
    """ ref and plus must be closer to each other than to minus """ 

    def getSample(self) : 
        n = len(self.svgdatas)
        dataPt = self.rng.choice(self.svgdatas) # choose random data point
        dataPt = self.transform(deepcopy(dataPt), self.svgdatas) # transform it
        docbox = getDocBBox(dataPt.doc)
        nodes = list(filterNodes(
            dataPt.nodes,
            complement(pathBBoxTooSmall),
            'bbox'
        ))
        nodes = list(filterNodes(
            dataPt.nodes, 
            lambda ps : len(ps) < self.opts.max_len, 
            'pathSet'
        ))
        try : 
            for attempt in range(10) : 
                three = self.rng.sample(nodes, k=3) # sample three nodes
                pairs = list(combinations(three, 2)) # make 3 pairs
                distances = list(starmap(
                    partial(distanceInTree, dataPt), 
                    pairs
                )) # compute distance in tree between pairs
                # find the permutation, if such exists such that
                # two nodes are closer to each other than the third node
                perm = next(
                    filter(
                        lambda p : (distances[p[0]] < distances[p[1]]) \
                                and (distances[p[0]] < distances[p[2]]), 
                        permutations([0, 1, 2], 3)
                    ), 
                    None
                ) 
                if perm is not None:  
                    ref = pairs[perm[0]][0]
                    plus = pairs[perm[0]][1]
                    minus = (set(three) - {ref, plus}).pop()
                    break
            # if unsuccessful, try sampling again
            if perm is None : 
                return self.getSample()
        except Exception as e : 
            return self.getSample()
        refMinus = distanceInTree(dataPt, ref, minus)
        refPlus = distanceInTree(dataPt, ref, plus)
        plusMinus = distanceInTree(dataPt, plus, minus)
        assert (refPlus < refMinus and refPlus < plusMinus)
        return (dataPt, ref, plus, minus, refPlus, refMinus)

