import random
from vectorrvnn.utils import *
from itertools import permutations, combinations, starmap
from functools import partial
from copy import deepcopy 

class TripletSampler () : 
    """ 
    Sample triplets (reference, positive, minus).

    Since each entry in a triplet is from the same 
    graphic, the output of the triplet sampler is
    a document for the graphic, indices of the 
    three entries and the scores of plus and minus with 
    respect to reference.

    This step can be done parallely. Then I'll collect the
    data points and indices and send it through a rasterizing
    phase which I believe can be done in reasonable time
    on a single process.

    Samplers API -
        1. Iterators API - __len__, __iter__, __next__
        2. reset() - reset the count and the random number 
                generator seed if needed.
    """

    def __init__ (self, svgdatas, length, opts, transform=lambda *args: args[0], val=False) :
        self.seed = rng.randint(0, 10000)
        self.rng = random.Random(self.seed) 
        self.svgdatas = svgdatas
        self.transform = transform 
        self.opts = opts
        self.val = val # In val mode the same triplets are sampled in each epoch.
        self.length = length
        self.i = 0

    def getSample (self) : 
        raise NotImplementedError

    def __iter__ (self): 
        return self

    def __next__ (self) : 
        if self.i < len(self) : 
            self.i += 1
            return self.getSample()
        else :
            self.i = 0
            if self.val : 
                self.rng = random.Random(self.seed)
            raise StopIteration

    def reset (self) : 
        self.i = 0
        if self.val : 
            self.rng = random.Random(self.seed)

    def __len__ (self) : 
        # Fixed number of samples for each epoch.
        return self.length

class SiblingSampler (TripletSampler) : 

    def getSample(self) : 
        n = len(self.svgdatas)
        while True : 
            dataPt = self.rng.choice(self.svgdatas) # choose random data point
            dataPt = self.transform(deepcopy(dataPt), self.svgdatas) # transform it
            found = False
            for attempt in range(10) : 
                try : 
                    ref = self.rng.choice(list(dataPt.nodes)) # set reference node
                    plus = self.rng.choice(list(siblings(dataPt, ref))) # set plus node
                    minus = self.rng.choice(list(
                        dataPt.nodes \
                                - descendants(
                                    dataPt, 
                                    parent(dataPt, ref)
                                )
                    )) # set minus node
                    found = True
                    break
                except Exception: 
                    pass
            if found : 
                break
        refMinus = lcaScore(dataPt, ref, minus)
        refPlus = lcaScore(dataPt, ref, plus)
        return (dataPt, ref, plus, minus, refPlus, refMinus)

class DiscriminativeSampler (TripletSampler):  
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

class AllSampler (TripletSampler):  

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
        try : 
            for attempt in range(10) : 
                three = self.rng.sample(nodes, k=3) # sample three nodes
                pairs = list(combinations(three, 2)) # make 3 pairs
                distances = list(starmap(
                    partial(distanceInTree, dataPt), 
                    pairs
                )) # compute distance in tree between pairs
                perm = next(
                    filter(
                        lambda p : (distances[p[0]] < distances[p[1]]),
                        permutations([0, 1, 2], 2)
                    ), 
                    None
                ) 
                if perm is not None:  
                    s0, s1 = set(pairs[perm[0]]), set(pairs[perm[1]])
                    ref = s0.intersection(s1).pop()
                    plus = (s0 - {ref}).pop()
                    minus = (set(three) - {ref, plus}).pop()
                    break
            # if unsuccessful, try sampling again
            if perm is None : 
                return self.getSample()
        except Exception as e : 
            return self.getSample()
        refMinus = distanceInTree(dataPt, ref, minus)
        refPlus = distanceInTree(dataPt, ref, plus)
        assert (refPlus < refMinus)
        return (dataPt, ref, plus, minus, refPlus, refMinus)

class LeafSampler (TripletSampler):  

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
        nodes = [n for n in nodes if dataPt.out_degree(n) == 0]
        try : 
            for attempt in range(10) : 
                three = self.rng.sample(nodes, k=3) # sample three nodes
                pairs = list(combinations(three, 2)) # make 3 pairs
                distances = list(starmap(
                    partial(distanceInTree, dataPt), 
                    pairs
                )) # compute distance in tree between pairs
                perm = next(
                    filter(
                        lambda p : (distances[p[0]] < distances[p[1]]),
                        permutations([0, 1, 2], 2)
                    ), 
                    None
                ) 
                if perm is not None:  
                    s0, s1 = set(pairs[perm[0]]), set(pairs[perm[1]])
                    ref = s0.intersection(s1).pop()
                    plus = (s0 - {ref}).pop()
                    minus = (set(three) - {ref, plus}).pop()
                    break
            # if unsuccessful, try sampling again
            if perm is None : 
                return self.getSample()
        except Exception as e : 
            return self.getSample()
        refMinus = distanceInTree(dataPt, ref, minus)
        refPlus = distanceInTree(dataPt, ref, plus)
        assert (refPlus < refMinus)
        return (dataPt, ref, plus, minus, refPlus, refMinus)

