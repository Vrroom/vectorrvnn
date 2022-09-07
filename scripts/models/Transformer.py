import torch 
from torch import nn
from torch.nn import functional as F
from vectorrvnn.utils import *
from vectorrvnn.trainutils import *
from vectorrvnn.network import *
from copy import deepcopy 
import math

def clones(module, N):
    """ Produce N identical layers. """
    return nn.ModuleList([deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    """ Encoder is made up of self-attn and feed forward """ 

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        """ mask helps mask out padded part of sequence """ 
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def attention(query, key, value, mask=None, dropout=None):
    """ 
    Compute 'Scaled Dot Product Attention' 
    
    Q, K, V are of size [B, H, L, D] 
        B = batch size
        H = number of heads
        L = sequence length
        D = embedding size
    """ 
    d_k = query.size(-1)
    # [B, H, L, L] should be symmetric
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) 
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # [B, H, L, L] 
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # [B, H, L, D]
    multiplied = torch.matmul(p_attn, value) 
    return multiplied, p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        """ Take in model size and number of heads. """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """ all inputs (except mask) are [B, L, D] """ 
        B = query.size(0)

        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(B, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    """ Implements FFN equation. """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class PositionalEncoding(nn.Module):
    """ Implement the PE function. """ 

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)

class Transformer(nn.Module):
    """ Core encoder is a stack of N layers """ 

    def __init__(self, layer, N):
        super(Transformer, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

def make_model(N=4, d_model=128, d_ff=512, h=8, dropout=0.1):
    """ Helper: Construct a model from hyperparameters. """ 
    c = deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    model = Transformer(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

class BBoxTransformer (TripletBase) : 
    def __init__(self, opts):  
        super(BBoxTransformer, self).__init__(opts)
        self.embed = nn.Linear(4, opts.embedding_size)
        for p in self.embed.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        self.net = make_model(d_model=opts.embedding_size) 

    def embedding (self, node, **kwargs) : 
        attn_mask = node['attn_mask'] # [B, L, L] 
        bboxes = node['bboxes'] # [B, L, 4] 
        embed = self.embed(bboxes) # [B, L, D] 
        encoded = self.net(embed, attn_mask) # [B, L, D] 
        # average only the relevant embeddings 
        avg_mask = node['avg_mask'] # [B, L, 1] 
        masked = (encoded * avg_mask).sum(1) # [B, D] 
        final = masked / avg_mask.sum(1) # [B, D] 
        return final

    @classmethod
    def nodeFeatures(cls, t, ps, opts) : 
        data_ = super(BBoxTransformer, cls).nodeFeatures(t, ps, opts)
        max_len = opts.max_len
        bbox = pathsetBox(t, ps)
        # combine actual boxes and dummy boxes
        bboxes = [t.bbox[i].tolist() for i in ps] + [[-1.0] * 4 for _ in range(max_len - len(ps))]
        data = dict()
        data['bboxes'] = torch.tensor(bboxes).float() 
        data['bbox'] = torch.tensor(bbox.tolist()).float()
        # prepare mask for dummy boxes (used for attn) 
        data['attn_mask'] = torch.zeros((max_len, max_len)).long() 
        data['attn_mask'][:len(ps), :len(ps)] = 1
        # prepare mask for averaging 
        data['avg_mask'] = torch.zeros((max_len, 1)).float()
        data['avg_mask'][:len(ps)] = 1
        return {**data, **data_}
