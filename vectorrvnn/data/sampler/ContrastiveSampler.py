import random
from vectorrvnn.utils import *
from itertools import permutations, combinations, starmap
from functools import partial
from .Sampler import *
from copy import deepcopy 

class ContrastiveSampler (Sampler):  
    """
    Give a three lists: 
        1. Pairs: (t, ps) 
        2. List of positive examples for each pair
        3. List of negative examples for each pair
    """
    def getSample (self) : 
        bs = self.opts.batch_size
        ts, ms, ps = [], [], []
        # TODO: Add this to options
        rs = self.opts.n_random_samples
        # easy way to shuffle data without deep copying
        ids = list(range(len(self.svgdatas)))
        self.rng.shuffle(ids)
        for i in ids : 
            total = len(ts) 
            if total > bs : break
            T = self.svgdatas[i]
            ps  = list(range(T.nPaths))
            # Add the random samples for negatives. 
            # TODO make size of random pathset a command line arg.
            for i in range(rs) : 
                rps = self.rng.sample(ps, k=min(len(ps), 3))
                ts.append((T, ps))
                ms.append([])
                ps.append([])

            for n in sorted(T.nodes) : 
                if n == findRoot(T) : continue
                ts.append((T, T.nodes[n]['pathSet']))
                S = siblings(T, n)
                p = [total + rs + _ for _ in S]
                m = [total + rs + _ for _ in (T.nodes - S)] 
                m += [total + i for i in range(rs)]
                ps = [_ for _ in ps if _ < bs]
                ms = [_ for _ in ps if _ < bs]
                ps.append(p)
                ms.append(m)

        # now remove the extra crap. 
        ts, ms, ps = ts[:bs], ms[:bs], ps[:bs]
        return ts, ms, ps
