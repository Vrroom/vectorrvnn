import torch 
from torch import nn
from torch.nn import functional as F
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase

class BBoxNet (TripletBase) : 
    def __init__(self, opts):  
        super(BBoxNet, self).__init__(opts)
        self.net = fcn(opts, 4, opts.embedding_size)

    def embedding (self, node, **kwargs) : 
        bbox = node['bbox']
        return self.net(bbox)

    @classmethod
    def nodeFeatures(cls, t, ps, opts) : 
        data = dict()
        data['tree'] = t
        data['pathSet'] = ps
        bbox = pathsetBox(t, ps)
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        return data
