import torch 
from torch import nn
from torch.nn import functional as F
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from .TripletBase import TripletBase

class PENet (TripletBase) : 

    def __init__ (self, opts) : 
        super(PENet, self).__init__(opts)
        # define the function approximators 
        # notation as per DeepSets:
        #   https://papers.nips.cc/paper/2017/file/f22e4747da1aa27e363d86d40ff442fe-Paper.pdf
        max_len = opts.max_len
        embedding_size = opts.embedding_size

        self.phi = fcn(opts, embedding_size, embedding_size)
        self.rho = fcn(opts, embedding_size, embedding_size)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, embedding_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2) *
                             -(math.log(10000.0) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a 0 pad at the end of pe so that unwanted indices
        # don't contribute to the sum.
        pe = torch.cat((pe, torch.zeros((1, embedding_size))), dim=0)
        self.register_buffer('pe', pe)

    def embedding (self, node, **kwargs) : 
        positions = node['positions']
        bs, max_len = positions.shape
        embedding_size = self.opts.embedding_size
        dim = max_len + 1
        positions = (positions + dim) % dim 
        pes = torch.index_select(self.pe, dim=0, index=positions.view(-1))
        fpes = self.phi(pes)
        fpes = fpes.view((bs, max_len, embedding_size))
        fpes = torch.sum(fpes, dim=1)
        spes = self.rho(fpes)
        return spes

    @classmethod
    def nodeFeatures(cls, t, ps, opts) : 
        data = dict()
        max_len = opts.max_len
        positions = torch.tensor([*ps, *[-1 for _ in range(max_len - len(ps))]])
        data['positions'] = positions
        return data
