import torch
from torch.nn import functional as F
from vectorrvnn.trainutils import *
from itertools import starmap, combinations
from more_itertools import collapse
from functools import lru_cache
import numpy as np

class TripletBase (nn.Module) :
    """
    Loss functions are named as *Loss. 

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
        self.vis = [
            DistanceHistogramCallback(
                frequency=opts.frequency,
                env=opts.name + "_distance"
            )
        ]
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
        refEmbed   = self.embedding(ref  , **kwargs) 
        plusEmbed  = self.embedding(plus , **kwargs) 
        minusEmbed = self.embedding(minus, **kwargs) 
        dplus  = l2(refEmbed, plusEmbed)
        dminus = l2(refEmbed, minusEmbed)
        return dplus, dminus

    def maxMarginLoss (self, ref, plus, minus, **kwargs): 
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

    def hardSemiHardMaxMarginLoss (self, ref, plus, minus, **kwargs) : 
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

    def tripletLoss (self, ref, plus, minus, **kwargs) : 
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

    def hardTripletLoss (self, ref, plus, minus, **kwargs)  :
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
            docbox = getDocBBox(t.doc)
            box1 = pathsetBox(t, ps1)
            box2 = pathsetBox(t, ps2)
            if pathBBoxTooSmall(box1, docbox)\
                    or pathBBoxTooSmall(box2, docbox) : 
                return torch.tensor(np.inf).to(self.opts.device)
            return self.sim_criteria(psEmbedding(ps1), psEmbedding(ps2))

        if subtrees is None : 
            subtrees = leaves(t)

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

        return treeFromNestedArray(subtrees)

    @classmethod
    def nodeFeatures(cls, t, ps1, opts) : 
        """ 
        The output should be a deep dict. See `dictOps.py`
        for what a deep dict is. The only permissible values in the 
        deep dict are np.ndarrays.
        """
        raise NotImplementedError
