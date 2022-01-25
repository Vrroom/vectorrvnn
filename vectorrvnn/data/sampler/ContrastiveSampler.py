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
        rs = self.opts.n_random_samples
        # easy way to shuffle data without deep copying
        ids = list(range(len(self.svgdatas)))
        self.rng.shuffle(ids)
        for i in ids : 
            total = len(ts) 
            if total > bs : break
            T = self.transform(deepcopy(self.svgdatas[i]), self.svgdatas)
            pathSet = list(range(T.nPaths))
            # Add the random samples for negatives. 
            # TODO make size of random pathset a command line arg.
            for i in range(rs) : 
                rps = self.rng.sample(pathSet, k=min(T.nPaths, 3))
                ts.append((T, rps))
                ms.append([])
                ps.append([])

            root = findRoot(T)
            nonRoot = sorted(T.nodes - {root})
            newIdx = dict(zip(nonRoot, range(total + rs, total + rs + len(nonRoot))))
            for n in nonRoot : 
                ts.append((T, T.nodes[n]['pathSet']))
                S = siblings(T, n)
                p = [newIdx[_] for _ in S]
                m = [newIdx[_] for _ in (T.nodes - (S | {n, root}))] 
                m += list(range(total, total + rs))
                p = [_ for _ in p if _ < bs]
                m = [_ for _ in m if _ < bs]
                ps.append(p)
                ms.append(m)

        # now remove the extra crap. 
        ts, ms, ps = ts[:bs], ms[:bs], ps[:bs]
        return ts, ms, ps
