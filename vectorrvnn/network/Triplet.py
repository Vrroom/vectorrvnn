import torch
from torch import nn
import torch.nn.functional as F
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
import vectorrvnn.trainutils.Constants as C
from .PositionalEncoding import PositionalEncoding
from functools import lru_cache
from itertools import starmap, combinations
import os
import os.path as osp
import numpy as np
from torchvision.models import resnet50
from more_itertools import collapse

def set_parameter_requires_grad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def convnet () : 
    # Load BAM! pretrained resnet50
    BASE_DIR = os.path.dirname(os.path.abspath(''))
    MODEL_DIR = os.path.join(BASE_DIR, 'vectorrvnn', 'results', 'bam_aug2')
    model = resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 20)
    state_dict = torch.load(os.path.join(MODEL_DIR, "epoch_15.pth"))
    model.load_state_dict(state_dict['model'])
    # Use the weights of the pretrained model to 
    # create weights for the new model.
    stitchedWts = model.fc.weight.repeat((C.embedding_size // 20 + 1, 1))[:C.embedding_size, :]
    stitchedBias = model.fc.bias.repeat(C.embedding_size // 20 + 1)[:C.embedding_size]
    model.fc = nn.Linear(2048, C.embedding_size)
    model.fc.weight.data = stitchedWts
    model.fc.bias.data = stitchedBias
    # Use the convolutional part of resnet for feature extraction 
    # only. Train only the fully connected layer on top.
    set_parameter_requires_grad(model, False)
    set_parameter_requires_grad(model.layer4, True)
    set_parameter_requires_grad(model.fc, True)
    # Make sure that parameters are floats 
    # And load model on cuda!
    model = model.float()
    model.to("cuda")
    return model

class TripletNet (nn.Module) :

    def __init__ (self, config) :
        super(TripletNet, self).__init__() 
        self.hidden_size = config['hidden_size']
        self.conv = convnet()
        self.pe = PositionalEncoding()
        self.nn = nn.Sequential(
            nn.Linear(3 * C.embedding_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, C.embedding_size)
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
        # TODO: Put a TopK based loss here!!
        refEmbed = self.embedding(im, refCrop, refWhole, refPosition)
        plusEmbed = self.embedding(im, plusCrop, plusWhole, plusPosition)
        minusEmbed = self.embedding(im, minusCrop, minusWhole, minusPosition)
        dplus  = torch.sum((plusEmbed  - refEmbed) ** 2, dim=1, keepdims=True)
        dminus = torch.sum((minusEmbed - refEmbed) ** 2, dim=1, keepdims=True)
        dplus_ = F.softmax(torch.cat((dplus, dminus), dim=1), dim=1)[:, 0]
        mask = dplus_ > 0.4
        hardRatio = mask.sum() / dplus.shape[0]
        dplus_ = dplus_[mask]
        dratio = (dminus[mask] / dplus[mask])
        dplus_ = dplus_ * refMinus[mask] / refPlus[mask]
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

