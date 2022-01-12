import torch
from torch.nn import functional as F
from vectorrvnn.trainutils import *
from vectorrvnn.geometry import *
from .EmbeddingBase import EmbeddingBase

class TripletBase (EmbeddingBase) :
    """
    Losses and forward pass for training
    neural networks using Triplet Learning. 

    Each loss function outputs a dictionary 
    with keys:
        1. loss    - self explanatory.
        2. mask    - which nodes were used. (default - None)
        3. dplus   - distance between ref and plus.
        4. dminus  - distance between ref and minus.
        5. hardpct - percentage of hard triplets.
    """
    def __init__ (self, opts) :
        super(TripletBase, self).__init__(opts)

    def _distances2Ref (self, ref, plus, minus, **kwargs) : 
        ref   = unitNorm(ref)
        plus  = unitNorm(plus)
        minus = unitNorm(minus)
        dplus  = l2(ref, plus)
        dminus = l2(ref, minus)
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
        ref   = unitNorm(ref)
        plus  = unitNorm(plus)
        minus = unitNorm(minus)
        # compute the cosine similarity and divide by temperature
        splus  = (ref * plus ).sum(dim=1, keepdim=True) / temperature
        sminus = (ref * minus).sum(dim=1, keepdim=True) / temperature
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
        sforward = super(TripletBase, self).forward
        ref   = sforward(ref)
        plus  = sforward(plus)
        minus = sforward(minus)
        lossFn = getattr(self, self.opts.loss)
        return lossFn(ref, plus, minus, **kwargs)

