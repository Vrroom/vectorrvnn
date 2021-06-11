import torch
from torch import nn
import torch.nn.functional as F
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from .PositionalEncoding import PositionalEncoding
from itertools import starmap, combinations
from more_itertools import collapse
import os
import os.path as osp
import numpy as np
from torchvision.models import *

def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def convnet (opts) : 
    model = resnet18(pretrained=True)
    # Use the weights of the pretrained model to 
    # create weights for the new model.
    inFeatures = model.fc.in_features
    model.fc = nn.Linear(inFeatures, opts.embedding_size)
    # Make sure that parameters are floats 
    # And load model on cuda!
    model = model.float()
    model.to("cuda")
    return model

class TripletNet (nn.Module) :

    def __init__ (self, opts) :
        super(TripletNet, self).__init__() 
        self.opts = opts
        self.conv = convnet(opts)
        self.pe = PositionalEncoding(opts)
        if opts.hidden_size is not None : 
            self.nn = nn.Sequential(
                nn.Linear(3 * opts.embedding_size, opts.hidden_size),
                nn.ReLU(),
                nn.Linear(opts.hidden_size, opts.embedding_size)
            )
        else : 
            self.nn = nn.Linear(
                3 * opts.embedding_size, 
                opts.embedding_size
            )

    def embedding (self, im, crop, whole, position) : 
        imEmbed = self.conv(im)
        cropEmbed = self.pe(self.conv(crop), position)
        wholeEmbed = self.pe(self.conv(whole), position)
        cat = torch.cat((imEmbed, cropEmbed, wholeEmbed), dim=1)
        return self.nn(cat)

    def forward (self, 
            im,
            refCrop, refWhole, refPosition,
            plusCrop, plusWhole, plusPosition,
            minusCrop, minusWhole, minusPosition,
            refPlus, refMinus) : 
        # Start with simple max-margin loss
        refEmbed = self.embedding(im, refCrop, refWhole, refPosition)
        plusEmbed = self.embedding(im, plusCrop, plusWhole, plusPosition)
        minusEmbed = self.embedding(im, minusCrop, minusWhole, minusPosition)
        dplus  = torch.sum((plusEmbed  - refEmbed) ** 2, dim=1, keepdims=True)
        dminus = torch.sum((minusEmbed - refEmbed) ** 2, dim=1, keepdims=True)
        margin = torch.relu(dplus - dminus + self.opts.max_margin)
        # FIXME: define the API for convenience and generality
        mask = (margin == margin).squeeze()
        hardRatio=mask.sum() / mask.nelement()
        dplus_ = margin
        dratio = (dminus[mask] / dplus[mask])
        # dplus_ = dplus_ * refMinus[mask] / refPlus[mask]
        return dict(dplus_=dplus_, dratio=dratio, hardRatio=hardRatio, mask=mask)

    def greedyTree (self, t, subtrees=None, binary=False) : 

        def distance (ps1, ps2) : 
            seenPathSets.add(asTuple(ps1))
            seenPathSets.add(asTuple(ps2))
            em1 = getEmbedding(t, ps1, self.embedding) 
            em2 = getEmbedding(t, ps2, self.embedding) 
            return torch.linalg.norm(em1 - em2)

        def subtreeEval (candidate) : 
            childPathSets = [tuple(collapse(c)) for c in candidate]
            return max(starmap(distance, combinations(childPathSets, 2)))

        def simplify (a, b) : 
            if binary : return (a, b)
            candidates = []
            candidatePatterns = ['(*a, *b)', '(a, b)', '(*a, b)', '(a, *b)']
            for pattern in candidatePatterns :
                try : 
                    candidates.append(eval(pattern))
                except Exception : 
                    pass
            scores = list(map(subtreeEval, candidates))
            best = candidates[argmin(scores)]
            return best

        if subtrees is None : 
            subtrees = leaves(t)
        seenPathSets = set()
        with torch.no_grad() : 
            while len(subtrees) > 1 : 
                treePairs = list(combinations(subtrees, 2))
                pathSets  = [tuple(collapse(s)) for s in subtrees]
                options   = list(combinations(pathSets, 2))
                distances = list(starmap(distance, options))
                left, right = treePairs[argmin(distances)]
                newSubtree = simplify(left, right)
                subtrees.remove(left)
                subtrees.remove(right)
                subtrees.append(newSubtree)

        return treeFromNestedArray(subtrees)

