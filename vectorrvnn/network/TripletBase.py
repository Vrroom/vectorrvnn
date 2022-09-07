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

    def ncs (self, ref, plus, minus, **kwargs) : 
        # verify and annotate dimensions
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
        hardpct = mask.sum() / float(mask.nelement())
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

