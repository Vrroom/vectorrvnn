import torch
from torch.nn import functional as F
from vectorrvnn.trainutils import *
from vectorrvnn.geometry import *
from vectorrvnn.baselines.autogroup import * 
from sklearn.cluster import AgglomerativeClustering
from itertools import starmap, combinations
from more_itertools import collapse
from functools import lru_cache
import numpy as np
from copy import deepcopy

class TripletBase (nn.Module) :
    """
    Each loss function outputs a dictionary 
    with keys:
        1. loss    - self explanatory.
        2. mask    - which nodes were used. (default - None)
        3. dplus   - distance between ref and plus.
        4. dminus  - distance between ref and minus.
        5. hardpct - percentage of hard triplets.
    """
    def __init__ (self, opts) :
        super(TripletBase, self).__init__() 
        self.opts = opts
        self.sim_criteria = globals()[opts.sim_criteria]

    def embedding (self, node, **kwargs) : 
        """ 
        Every subclass comes with its own implementation
        of this. This seperation makes it easier to write
        common loss functions that I might want to use
        """
        raise NotImplementedError

    def _distances2Ref (self, ref, plus, minus, **kwargs) : 
        refEmbed   = unitNorm(self.embedding(ref  , **kwargs))
        plusEmbed  = unitNorm(self.embedding(plus , **kwargs))
        minusEmbed = unitNorm(self.embedding(minus, **kwargs))
        dplus  = l2(refEmbed, plusEmbed)
        dminus = l2(refEmbed, minusEmbed)
        return dplus, dminus

    def maxMargin (self, ref, plus, minus, **kwargs): 
        maxMargin = self.opts.max_margin
        assert(maxMargin is not None)
        dplus, dminus = self._distances2Ref(ref, plus, minus, **kwargs)
        margin = torch.relu(dplus - dminus + maxMargin)
        mask = (dplus >= dminus).view(-1, 1)
        hardpct = mask.sum() / mask.nelement()
        loss = margin.mean()
        return dict(
            loss=loss,
            mask=None,
            dplus=dplus,
            dminus=dminus,
            hardpct=hardpct
        )

    def hardMaxMargin (self, ref, plus, minus, **kwargs) : 
        """
        From FaceNet: A Unified Embedding for Face Recognition and Clustering
        """
        maxMargin = self.opts.max_margin
        assert(maxMargin is not None)
        dplus, dminus = self._distances2Ref(ref, plus, minus, **kwargs)
        margin = torch.relu(dplus - dminus + maxMargin)
        mask = (dplus >= dminus).view(-1, 1)
        loss = maskedMean(margin, mask)
        hardpct = mask.sum() / mask.nelement()
        return dict(
            loss=loss,
            mask=mask,
            dplus=dplus,
            dminus=dminus,
            hardpct=hardpct
        )

    def triplet (self, ref, plus, minus, **kwargs) : 
        """ 
        Triplet loss defined in the original triplet learning paper: 
            Deep Metric Learning Using Triplet Network
        """
        hardThreshold = self.opts.hard_threshold
        assert (hardThreshold is not None)
        dplus, dminus = self._distances2Ref(ref, plus, minus, **kwargs)
        cre = F.softmax(
            torch.cat((dplus, dminus), dim=1), 
            dim=1
        )
        loss = (cre ** 2).mean()
        mask = (cre > hardThreshold).view(-1, 1)
        hardpct = mask.sum() / mask.nelement()
        return dict(
            loss=loss,
            mask=None,
            dplus=dplus,
            dminus=dminus,
            hardpct=hardpct
        )

    def hardTriplet (self, ref, plus, minus, **kwargs)  :
        hardThreshold = self.opts.hard_threshold
        assert (hardThreshold is not None)
        dplus, dminus = self._distances2Ref(ref, plus, minus, **kwargs)
        cre = F.softmax(
            torch.cat((dplus, dminus), dim=1), 
            dim=1
        )
        mask = (cre > hardThreshold).view(-1, 1)
        loss = maskedMean(cre ** 2, mask)
        hardpct = mask.sum() / mask.nelement()
        return dict(
            loss=loss,
            mask=mask,
            dplus=dplus,
            dminus=dminus,
            hardpct=hardpct
        )

    def infoNCE (self, ref, plus, minus, **kwargs) : 
        temperature = self.opts.temperature
        assert (temperature is not None)
        # Find and normalize embeddings
        refEmbed   = unitNorm(self.embedding(ref  , **kwargs))
        plusEmbed  = unitNorm(self.embedding(plus , **kwargs))
        minusEmbed = unitNorm(self.embedding(minus, **kwargs))
        # compute the cosine similarity and divide by temperature
        splus  = (refEmbed * plusEmbed ).sum(dim=1, keepdim=True) / temperature
        sminus = (refEmbed * minusEmbed).sum(dim=1, keepdim=True) / temperature
        # compute loss, mask and hardpct.
        two = torch.cat((splus, sminus), dim=1)
        exp = torch.softmax(two, dim=1)[:, 0]
        loss = -torch.log(exp).mean()
        mask = (splus < sminus).view(-1, 1)
        hardpct = mask.sum() / mask.nelement()
        dminus = (1 / temperature) - sminus
        dplus  = (1 / temperature) - splus
        return dict(
            loss=loss,
            mask=None,
            dplus=dplus,
            dminus=dminus,
            hardpct=hardpct
        )

    def forward (self, ref, plus, minus, **kwargs) : 
        # figure out which loss to use from opts.
        lossFn = getattr(self, self.opts.loss)
        return lossFn(ref, plus, minus, **kwargs)

    def greedyTree (self, t, subtrees=None) : 

        @lru_cache(maxsize=128)
        def psEmbedding (ps) : 
            f = self.nodeFeatures(t, ps, self.opts)
            tensorApply(
                f, 
                lambda x : x.to(self.opts.device).unsqueeze(0)
            )
            return self.embedding(f)

        def distance (ps1, ps2) : 
            box1 = pathsetBox(t, ps1)
            box2 = pathsetBox(t, ps2)
            if pathBBoxTooSmall(box1) or pathBBoxTooSmall(box2) : 
                return torch.tensor(np.inf).to(self.opts.device)
            return self.sim_criteria(psEmbedding(ps1), psEmbedding(ps2))

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

        @lru_cache(maxsize=128)
        def psEmbedding (ps) : 
            f = self.nodeFeatures(t, ps, self.opts)
            tensorApply(
                f, 
                lambda x : x.to(self.opts.device).unsqueeze(0)
            )
            return self.embedding(f)

        self.eval()
        with torch.no_grad() :
            embeddings = [unitNorm(psEmbedding((i,))) for i in subtrees]
            stacked = torch.cat(embeddings).detach().cpu().numpy()

        cpy = deepcopy(t)

        if len(subtrees) == 1 : 
            cpy.initTree(hac2nxDiGraph(subtrees, []))
        else : 
            agg = AgglomerativeClustering(1, affinity='cosine', linkage='single')
            agg.fit(stacked)
            cpy.initTree(hac2nxDiGraph(subtrees, agg.children_))
        return cpy
    
    def containmentGuidedTreeHac(self, t, subtrees=None): 
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
            trees[p] = nx.DiGraph(self.hacTree(t, subtrees=x))
            trees[p] = nx.relabel_nodes(trees[p], dict(map(reversed, enumerate(x))))

        nestedArray = containmentMerge(n, containmentGraph, trees)[1]
        cpy = deepcopy(t)
        cpy.initTree(parentheses2tree(nestedArray))
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
            trees[p] = nx.relabel_nodes(trees[p], dict(map(reversed, enumerate(x))))

        nestedArray = containmentMerge(n, containmentGraph, trees)[1]
        cpy = deepcopy(t)
        cpy.initTree(parentheses2tree(nestedArray))
        return cpy

    @classmethod
    def nodeFeatures(cls, t, ps, opts) : 
        """ 
        The output should be a deep dict. See `dictOps.py`
        for what a deep dict is. The only permissible values in the 
        deep dict are np.ndarrays.
        """
        raise NotImplementedError
