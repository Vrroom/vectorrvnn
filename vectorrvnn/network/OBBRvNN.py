import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase

class OBBRvNN (TripletBase) : 

    def __init__ (self, opts) :
        super(OBBRvNN, self).__init__(opts) 
        self.embedder = fcn(opts, 10, opts.embedding_size)
        module = fcn(opts, opts.embedding_size, opts.embedding_size)
        self.phi = deepcopy(module)
        self.rho = deepcopy(module)
        self.projector = deepcopy(module)

    def treeEncode(self, T, r, obbs, ps) : 
        if T.out_degree(r) == 0 : 
            i = ps.index(r)
            return self.embedder(obbs[i].view((1, -1)))
        else : 
            children = []
            for c in T.neighbors(r) : 
                children.append(self.treeEncode(T, c, obbs, ps))
            F = torch.cat(children, dim=0)
            F = self.phi(F)
            return self.rho(F.sum(0, keepdim=True))

    def embedding (self, node, **kwargs) : 
        tree = node['tree'][0]
        obbs = node['obb'].squeeze(0)
        r = node['n'][0]
        ps = node['pathSet'][0]
        root = self.treeEncode(tree, r, obbs, ps)
        return self.projector(root)

    def greedyTree (self, t, subtrees=None) : 

        @lru_cache(maxsize=128)
        def psEmbedding (tree) : 
            cpy = deepcopy(t)
            cpy.initTree(parentheses2tree(tree))
            ps = cpy.nodes[findRoot(cpy)]['pathSet']
            f = self.nodeFeatures(cpy, ps, self.opts)
            tensorApply(
                f, 
                lambda x : x.to(self.opts.device).unsqueeze(0)
            )
            for k in f.keys() : 
                if not isinstance(f[k], torch.Tensor) : 
                    f[k] = [f[k]]
            return self.embedding(f)

        def distance (t1, t2) : 
            ps1, ps2 = list(collapse(t1)), list(collapse(t2))
            box1 = pathsetBox(t, ps1)
            box2 = pathsetBox(t, ps2)
            if pathBBoxTooSmall(box1) or pathBBoxTooSmall(box2) : 
                return torch.tensor(np.inf).to(self.opts.device)
            return self.sim_criteria(psEmbedding(t1), psEmbedding(t2))

        if subtrees is None : 
            subtrees = leaves(t)
        subtrees = deepcopy(subtrees)

        self.eval()
        with torch.no_grad() : 
            while len(subtrees) > 1 : 
                treePairs = list(combinations(subtrees, 2))
                distances = list(starmap(distance, treePairs))
                left, right = treePairs[argmin(distances)]
                newSubtree = (left, right)
                subtrees.remove(left)
                subtrees.remove(right)
                subtrees.append(newSubtree)

        cpy = deepcopy(t)
        cpy.initTree(parentheses2tree(subtrees[0]))
        return cpy

    @classmethod 
    def nodeFeatures (cls, t, ps, opts): 
        ps = sorted(ps)
        data = dict()
        data['tree'] = t
        data['pathSet'] = ps
        data['n'] = next(filterNodes(
            t.nodes, 
            lambda ps_ : ps == ps_, 
            'pathSet'
        ))
        obbs = [t.obbs[i].tolist() for i in ps]
        data['obb'] = torch.tensor(obbs).float()
        return data


