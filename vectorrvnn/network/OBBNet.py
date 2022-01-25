"""
Modules in this file are 
based on minor modifications on: 
    https://nlp.seas.harvard.edu/2018/04/03/attention.html
"""
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase
from .ContrastiveBase import ContrastiveBase

class Encoder (nn.Module) : 

    def __init__ (self, layer, N) :
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward (self, x) :
        for layer in self.layers :
            x = layer(x)
        return self.norm(x)

class SublayerConnection (nn.Module) :

    def __init__ (self, size, dropout) :
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward (self, x, sublayer) : 
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer (nn.Module) :

    def __init__(self, size, sAttn, ff, dropout) : 
        super(EncoderLayer, self).__init__()
        self.sAttn = sAttn
        self.ff = ff
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x) :
        x = self.sublayer[0](x, lambda x: self.sAttn(x, x, x)[0])
        x = self.sublayer[1](x, self.ff)
        return x

class OBBNet (ContrastiveBase) : 

    def __init__ (self, opts) :
        super(OBBNet, self).__init__(opts) 
        c = deepcopy
        attn = nn.MultiheadAttention(opts.embedding_size, opts.heads, opts.dropout)
        ff = fcn(opts, opts.embedding_size, opts.embedding_size)
        self.embedder =  nn.Linear(16, opts.embedding_size)
        self.encoder = Encoder(
            EncoderLayer(
                opts.embedding_size, 
                deepcopy(attn),
                deepcopy(ff),
                opts.dropout
            ), 
            opts.encoder_layers
        )
        self.apply(getInitializer(opts))

    def embedding (self, node, **kwargs) : 
        obb, fill, stroke = node['obb'], node['fill'], node['stroke']
        f = torch.cat((obb, fill, stroke), 2)
        f = self.embedder(f)
        f = self.encoder(f)
        f = f.mean(1)
        return f

    @classmethod 
    def nodeFeatures (cls, t, ps, opts): 
        data_ = super(OBBNet, cls).nodeFeatures(t, ps, opts)
        obbs    = np.array([t.obbs[i].tolist() for i in ps])
        fills   = np.array([t.fills[i] for i in ps])
        strokes = np.array([t.strokes[i] for i in ps])
        data = dict(
            obb=obbs,
            fill=fills,
            stroke=strokes
        )
        tensorApply(data, lambda x : torch.tensor(x).float(), module=np)
        return { **data, **data_ }

