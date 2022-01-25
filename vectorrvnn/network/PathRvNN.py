import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
import numpy as np
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .RvNNBase import *

class PathRvNN (RvNNBase) : 

    def __init__ (self, opts) :
        super(PathRvNN, self).__init__(opts) 
        self.pathembedder = nn.Sequential(
            nn.Linear(16, opts.embedding_size),
            nn.Sigmoid()
        )
        self.scorer = nn.Linear(opts.embedding_size, 1, bias=False)
        self.combiner = nn.Sequential(
            nn.Linear(2 * opts.embedding_size, opts.embedding_size),
            nn.Sigmoid()
        )
        self.apply(getInitializer(opts))
    
    def embedding (self, node, **kwargs) : 
        obb, fill, stroke = node['obb'], node['fill'], node['stroke']
        f = torch.cat((obb, fill, stroke), 2).squeeze(0)
        return self.pathembedder(f)

    def groupEmbedding (self, f1, f2, **kwargs) : 
        return self.combiner(torch.cat((f1, f2), 1))

    def score (self, feature) : 
        return self.scorer(feature)

    @classmethod 
    def nodeFeatures (cls, t, ps, opts): 
        data_ = super(PathRvNN, cls).nodeFeatures(t, ps, opts)
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



