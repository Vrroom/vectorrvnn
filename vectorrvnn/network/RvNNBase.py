import torch
from torch.nn import functional as F
from vectorrvnn.trainutils import *
from vectorrvnn.geometry import *
from vectorrvnn.utils import *
from .EmbeddingBase import *
from itertools import starmap, permutations
from functools import lru_cache
from copy import deepcopy

class RvNNBase (EmbeddingBase) : 
    """ 
    Base class for learning through the 
    structured SVM method using by Socher et. al. for 
    sentence/image parsing. 
    """
    def __init__ (self, opts) :
        super(EmbeddingBase, self).__init__() 
        self.opts = opts
        self.sim_criteria = globals()[opts.sim_criteria]
    
    def embedding (self, node, **kwargs) : 
        """ To be used only for leaf nodes """ 
        raise NotImplementedError

    def groupEmbedding (self, f1, f2, **kwargs) : 
        raise NotImplementedError

    def score (self, feature) : 
        raise NotImplementedError

    def structuredHinge (self, ts, ts_) :  
        kappa = self.opts.kappa
        assert(kappa is not None)
        gamma = self.opts.wd
        totalRisk = 0
        for t, t_ in zip(ts, ts_) : 
            fs  = [self.subtreeFeature2(t , n) for n in nonLeaves(t )]
            fs_ = [self.subtreeFeature2(t_, n) for n in nonLeaves(t_)]
            s  = sum(self.score(f) for f in fs ) 
            s_ = sum(self.score(f) for f in fs_) 
            margin = kappa * cted(t, t_) 
            risk = s_ - s + margin
            totalRisk += risk
        wtSqNorms = [(x ** 2).sum() for x in self.parameters()]
        wtLoss = sum(wtSqNorms) * gamma / 2
        loss = (totalRisk / len(ts)) + wtLoss
        return dict(
            loss=loss,
            mask=None,
            dplus=None,
            dminus=None,
            hardpct=None
        )

    @lru_cache(maxsize=1000)
    def subtreeFeature1 (self, T, subtree): 
        if isinstance(subtree, int): 
            return self.nodeEmbedding(T, subtree)
        l, r = subtree
        f1, f2 = self.subtreeFeature1(T, l), self.subtreeFeature1(T, r)
        return self.groupEmbedding(f1, f2) 

    @lru_cache(maxsize=1000)
    def subtreeFeature2(self, T, n) : 
        if T.out_degree(n) == 0 : 
            return self.nodeEmbedding(T, n)
        l, r = T.neighbors(n)
        f1, f2 = self.subtreeFeature2(T, l), self.subtreeFeature2(T, r)
        return self.groupEmbedding(f1, f2) 
            
    def forward (self, ts, **kwargs) : 
        ts_ = [self.greedyTree(t) for t in ts]     
        lossFn = getattr(self, self.opts.loss)
        return lossFn(ts, ts_, **kwargs)
    
    def greedyTree (self, t, subtrees=None) : 

        if subtrees is None : 
            subtrees = leaves(t)
        subtrees = deepcopy(subtrees)

        self.eval()
        with torch.no_grad() : 
            while len(subtrees) > 1 : 
                pairs = list(permutations(subtrees, 2))
                features = [self.subtreeFeature1(t, st) for st in pairs]
                scores = list(map(self.score, features))
                left, right = pairs[argmax(scores)]
                newSubtree = (left, right)
                subtrees.remove(left)
                subtrees.remove(right)
                subtrees.append(newSubtree)
        self.train()
        cpy = deepcopy(t)
        cpy.initTree(parentheses2tree(subtrees[0]))
        return cpy

    def nodeEmbedding(self, T, n) :
        ps = tuple(T.nodes[n]['pathSet'])
        return self.pathSetEmbedding(T, ps)

    @lru_cache(maxsize=1024)
    def pathSetEmbedding(self, T, ps) :
        fn = self.nodeFeatures(T, ps, self.opts)
        tensorApply(
            fn,
            lambda x : x.to(self.opts.device).unsqueeze(0)
        )
        return self.embedding(fn)
