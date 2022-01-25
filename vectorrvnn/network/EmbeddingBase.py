import torch
from torch.nn import functional as F
from vectorrvnn.trainutils import *
from vectorrvnn.geometry import *
from vectorrvnn.utils import *
from vectorrvnn.baselines.autogroup import * 
from sklearn.cluster import AgglomerativeClustering
from itertools import starmap, combinations
from more_itertools import collapse
from functools import lru_cache
import numpy as np
from copy import deepcopy

_affinity = dict(
    ncs='cosine',
    l2='euclidean'
)

class EmbeddingBase (nn.Module) :
    """ 
    Base class for neural networks that learn
    through contrastive learning.
    
    For our purpose, embeddings are used to compute
    trees. This class provides operations for 3 strategies
    for tree construction:

        1. greedyTree: greedily group pairs of subgroups
           based on similarity of embeddings.
        2. hacTree: compute leaf embeddings and do 
           hierarchical agglomerative clustering. 
        3. containmentGuidedTree: construct tree greedily
           respecting containment constraints.
    """
    def __init__ (self, opts) :
        super(EmbeddingBase, self).__init__() 
        self.opts = opts
        self.sim_criteria = globals()[opts.sim_criteria]

    def embedding (self, node, **kwargs) : 
        """ 
        Every subclass comes with its own implementation
        of this. This seperation makes it easier to write
        common loss functions that I might want to use
        """
        raise NotImplementedError

    def forward (self, batch, **kwargs) : 
        es = []
        for block in batch : 
            es.append(self.embedding(block))
        return torch.cat(es, 0)

    def greedyTree (self, t, subtrees=None) : 

        def distance (ps1, ps2) : 
            box1 = pathsetBox(t, ps1)
            box2 = pathsetBox(t, ps2)
            if pathBBoxTooSmall(box1) or pathBBoxTooSmall(box2) : 
                return torch.tensor(np.inf).to(self.opts.device)
            return self.sim_criteria(
                unitNorm(self.pathSetEmbedding(t, ps1)),
                unitNorm(self.pathSetEmbedding(t, ps2))
            )

        if subtrees is None : 
            subtrees = leaves(t)
        subtrees = deepcopy(subtrees)

        self.eval()
        with torch.no_grad() : 
            while len(subtrees) > 1 : 
                treePairs = list(combinations(subtrees, 2))
                pathSets  = [tuple(collapse(s)) for s in subtrees]
                options   = list(combinations(pathSets, 2))
                distances = list(starmap(distance, options))
                left, right = treePairs[argmin(distances)]
                newSubtree = (left, right)
                subtrees.remove(left)
                subtrees.remove(right)
                subtrees.append(newSubtree)

        cpy = deepcopy(t)
        cpy.initTree(parentheses2tree(subtrees[0]))
        return cpy

    def hacTree (self, t, subtrees=None) : 
        if subtrees is None : 
            subtrees = leaves(t)

        self.eval()
        with torch.no_grad() :
            embeddings = [self.pathSetEmbedding(t, (i,)) for i in subtrees]
            embeddings = [unitNorm(e) for e in embeddings]
            stacked = toNumpyCPU(torch.cat(embeddings))

        cpy = deepcopy(t)

        if len(subtrees) == 1 : 
            cpy.initTree(hac2nxDiGraph(subtrees, []))
        else : 
            agg = AgglomerativeClustering(
                1, 
                affinity=_affinity[self.opts.sim_criteria], 
                linkage='single'
            )
            agg.fit(stacked)
            cpy.initTree(hac2nxDiGraph(subtrees, agg.children_))
        return cpy

    def containmentGuidedTree (self, t, subtrees=None) : 
        if subtrees is None : 
            subtrees = leaves(t)

        n = t.nPaths
        containmentGraph = dropExtraParents(
            subgraph(
                relationshipGraph(
                    t.doc, 
                    bitmapContains, 
                    False,
                    threadLocal=self.opts.rasterize_thread_local
                ),
                lambda x: x['bitmapContains']
            )
        )
        containmentGraph.add_edges_from(
            [(n, _) for _ in containmentGraph.nodes 
                if containmentGraph.in_degree(_) == 0])

        parents = nonLeaves(containmentGraph)
        siblingSets = [list(containmentGraph.neighbors(p))
                for p in parents]

        trees = dict()
        for p, x in zip(parents, siblingSets):  
            trees[p] = nx.DiGraph(self.greedyTree(t, subtrees=x))
            newLabels = serialMapping(x)
            trees[p] = nx.relabel_nodes(trees[p], newLabels)

        nestedArray = containmentMerge(n, containmentGraph, trees)[1]
        cpy = deepcopy(t)
        cpy.initTree(parentheses2tree(nestedArray))
        return cpy

    @classmethod
    def nodeFeatures(cls, t, ps, opts) : 
        """ 
        The output should be a deep dict. See `dictOps.py`
        for what a deep dict is. 
        """
        data = dict()
        data['tree'] = t
        data['pathSet'] = sorted(ps)
        return data 

    def nodeEmbedding(self, T, n) :
        ps = tuple(T.nodes[n]['pathSet'])
        return self.pathSetEmbedding(T, ps)

    @lru_cache(maxsize=1024)
    def pathSetEmbedding(self, T, ps) :
        with torch.no_grad() :
            fn = self.nodeFeatures(T, ps, self.opts)
            tensorApply(
                fn,
                lambda x : x.to(self.opts.device).unsqueeze(0)
            )
            return self.embedding(fn)
