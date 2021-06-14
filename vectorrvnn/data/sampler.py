import random
from vectorrvnn.utils.graph import *

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

    def __init__ (self, svgdatas, length, transform=lambda *args: args[0], seed=0, val=False) :
        self.rng = random.Random(seed) 
        self.svgdatas = svgdatas
        self.transform = transform 
        self.seed = seed
        self.val = val # In val mode the same triplets are sampled in each epoch.
        self.length = length
        self.i = 0

    def getSample (self) : 
        n = len(self.svgdatas)
        try : 
            dataPt = self.rng.choice(self.svgdatas) # choose random data point
            dataPt = self.transform(dataPt, self.svgdatas) # transform it
            ref = self.rng.choice(list(dataPt.nodes)) # set reference nodes
            plus, minus = self.rng.sample(dataPt.nodes - [ref], k=2) # sample two other nodes
            # make 10 attempts to sample two nodes where one is closer to ref than other
            attempts = 0
            while lcaScore(dataPt, ref, plus) == lcaScore(dataPt, ref, minus) \
                    and attempts < 10:
                attempts += 1
                plus, minus = self.rng.sample(dataPt.nodes - [ref], k=2)
            # if unsuccessful, try sampling again
            if attempts >= 10 : 
                return self.getSample()
            # swap the order of plus and minus if required
            if lcaScore(dataPt, ref, plus) > lcaScore(dataPt, ref, minus) :
                minus, plus = plus, minus
        except Exception as e : 
            # exception will typically arise if you try to 
            # sample more elements than there are in the list
            return self.getSample()
        refMinus = lcaScore(dataPt, ref, minus)
        refPlus = lcaScore(dataPt, ref, plus)
        return (dataPt, ref, plus, minus, refPlus, refMinus)

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
