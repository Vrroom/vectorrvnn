import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase

class OBBGRU (TripletBase) : 

    def __init__ (self, opts) :
        super(OBBGRU, self).__init__(opts) 
        self.gru = nn.GRU(
            input_size=10, 
            hidden_size=opts.embedding_size,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=opts.dropout,
            bidirectional=True
        )
        self.projector = fcn(
            opts,
            2 * opts.embedding_size,
            opts.embedding_size
        )
        self.apply(getInitializer(opts))

    def embedding (self, node, **kwargs) : 
        obb = node['obb']   
        f = self.gru(obb)[1]
        f = f.permute((1, 0, 2)).reshape((1, -1))
        f = self.projector(f)
        return f

    @classmethod 
    def nodeFeatures (cls, t, ps, opts): 
        ps = sorted(ps)
        data = dict()
        data['tree'] = t
        data['pathSet'] = ps
        obbs = [t.obbs[i].tolist() for i in ps]
        data['obb'] = torch.tensor(obbs).float()
        return data

