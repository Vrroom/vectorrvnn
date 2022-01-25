import torch
from torch.nn import functional as F
from vectorrvnn.trainutils import *
from vectorrvnn.geometry import *
from .EmbeddingBase import EmbeddingBase

class ContrastiveBase (EmbeddingBase) : 

    def __init__ (self, opts) : 
        super(ContrastiveBase, self).__init__(opts)

    def supCon (self, es, ps, ms, **kwargs) : 
        """ 
        Implementation of a contrastive loss from: 
            Supervised Contrastive Learning
            
        es is a batch of embeddings, ps are indices
        of positive samples and ms are the indices
        of the negative samples.
        """
        # Start with stupid implementation then try loop unrolling. 
        temperature = self.opts.temperature
        es = unitNorm(es) # batch_size by embedding_size
        sims = (es @ es.t()) / temperature
        loss, hardpct = 0, 0
        dplus, dminus = [], []
        for i, (p, m) in enumerate(zip(ps, ms)) : 
            msz = min(p.size(0), m.size(0))
            if msz == 0 : continue
            p_, m_ = sims[i, p], sims[i, m]
            logits = torch.cat((p_, m_))
            cre = -torch.log(torch.softmax(logits, 0))
            hardpct += lte(p_, m_).float().mean()
            loss  += cre[:p.size(0)].mean()
            dplus.append(-p_[:msz])
            dminus.append(-m_[:msz])
        loss /= self.opts.batch_size
        hardpct /= self.opts.batch_size
        dplus = torch.cat(dplus)
        dminus = torch.cat(dminus)
        return dict(
            loss=loss,
            mask=None, 
            dplus=dplus,
            dminus=dminus,
            hardpct=hardpct
        )

    def forward (self, nodes, ps, ms, **kwargs) : 
        sforward = super(ContrastiveBase, self).forward
        es = sforward(nodes)
        lossFn = getattr(self, self.opts.loss)
        return lossFn(es, ps, ms, **kwargs)
        
