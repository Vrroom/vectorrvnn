import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase
from .ContrastiveBase import ContrastiveBase

class PathGRU (ContrastiveBase) : 

    def __init__ (self, opts) :
        super(PathGRU, self).__init__(opts) 
        self.encode = nn.Linear(16, opts.embedding_size)
        self.gru = nn.GRU(
            input_size=opts.embedding_size, 
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
        obb, fill, stroke = node['obb'], node['fill'], node['stroke']
        f = torch.cat((obb, fill, stroke), 2)
        f = self.encode(f)
        f = self.gru(f)[1]
        f = f.permute((1, 0, 2)).reshape((1, -1))
        f = self.projector(f)
        return f

    @classmethod 
    def nodeFeatures (cls, t, ps, opts): 
        data_ = super(PathGRU, cls).nodeFeatures(t, ps, opts)
        ps = sorted(ps)
        obbs    = np.array([t.obbs[i].tolist() for i in ps])
        fills   = np.array([t.fills[i] for i in ps])
        strokes = np.array([t.strokes[i] for i in ps])
        data = dict(
            obb=obbs,
            fill=fills,
            stroke=strokes
        )
        tensorApply(data, lambda x : torch.tensor(x).float(), module = np)
        return {**data, **data_}

