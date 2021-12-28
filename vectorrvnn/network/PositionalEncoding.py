import torch
import torch.nn as nn
import math
from torch.autograd import Variable

# Source - http://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, opts, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        max_len = opts.max_len
        d_model = opts.embedding_size
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Add a 0 pad at the end of pe.
        pe = torch.cat((pe, torch.zeros((1, d_model))), dim=0)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, positions):
        """
        Parameters
        ----------
            x : torch.Tensor
                Shape - [bs, d_model]
            positions : torch.LongTensor
                Shape - [bs, max_len] 
        """
        # The positions matrix contains the zIndex orders of paths in each group.
        # x has bs number of groups in it. 
        # Since each group has variable number of paths, to make sure that 
        # positions matrix is fixed shape, -1 padding is added. This refers to
        # the last row of the positional encoding which is a row of 0's.
        # Because positional encoding for a group is the sum of positional encodings
        # of the paths in the group, this convention is very convenient.
        bs, d_model = x.shape
        _, max_len = positions.shape
        positions = (positions + self.pe.shape[1]) % self.pe.shape[1]
        relevantPes = torch.index_select(self.pe, dim=1, index=positions.view(-1)) # [1, bs * max_len, d_model]
        relevantPes = relevantPes.view((bs, max_len, d_model)) # [bs, max_len, d_model]
        relevantPes = torch.sum(relevantPes, dim=1) # [bs, d_model]
        x = x + Variable(relevantPes, requires_grad=False)
        return self.dropout(x)
