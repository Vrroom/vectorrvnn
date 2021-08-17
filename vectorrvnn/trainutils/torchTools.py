import torch
import torch.nn as nn
from more_itertools import flatten
from functools import partial
import numpy as np
from torchvision.models import *
from .initializer import getInitializer
from torchvision.ops import *

def tensorApply (thing, fn, 
    predicate=lambda x: True, module=torch) : 
    """ 
    Apply a tensor transformation to a tensor 
    or dictionary or a list.
    """
    Cls = torch.Tensor if module == torch else np.ndarray
    if isinstance(thing, Cls) and predicate(thing): 
        thing = fn(thing)
    elif isinstance(thing, dict) : 
        for k, v in thing.items() : 
            thing[k] = tensorApply(v, fn, predicate, module)
    elif isinstance(thing, list) : 
        for i, _ in enumerate(thing) : 
            thing[i] = tensorApply(thing[i], fn, 
                    predicate, module)
    return thing

def tensorFilter(thing, predicate, module=torch) :
    """
    Filter tensors from list/dictionary or
    just an individual collection based on whether 
    it passes a predicate. Returns a list.
    """
    Cls = torch.Tensor if module == torch else np.ndarray
    if isinstance(thing, Cls) and predicate(thing): 
        return [thing]
    elif isinstance(thing, dict) : 
        return list(flatten(map(
            partial(
                tensorFilter, 
                predicate=predicate,
                module=module
            ), 
            map(
                lambda k : thing[k],
                sorted(thing.keys())
            )
        )))
    elif isinstance(thing, list) : 
        return list(flatten(map(
            partial(
                tensorFilter, 
                predicate=predicate,
                module=module
            ), 
            thing
        )))
    return []

def setParameterRequiresGrad(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad

def l2 (a, b, eps=1e-5) : 
    """ 
    a.shape == b.shape == [B, N].
    B is the batch size and N is the dimension
    of the embedding.
    """
    d2 = torch.sum((a - b) ** 2, dim=1, keepdims=True)
    l2 = torch.sqrt(d2 + eps)
    return l2

def unitNorm(a) : 
    return a / a.norm(dim=1, keepdim=True)

def negativeCosineSimilarity (a, b) : 
    """
    a.shape == b.shape == [B, N].
    B is the batch size and N is the dimension
    of the embedding.
    """
    return - (unitNorm(a) * unitNorm(b)).sum(dim=1, keepdim=True)

def maskedMean(thing, mask, eps=1e-5) : 
    """
    thing.shape == mask.shape == [B, 1]
    Used to handle cases where nothing is 
    in the thing after mask is applied
    """
    return thing[mask].sum() / (mask.sum() + eps)

def normalize2UnitRange (thing) :
    m, M = thing.min(), thing.max()
    if m == M : 
        return torch.ones_like(thing)
    else : 
        return (thing - m) / (M - m)

def channelDim (thing, module=torch) : 
    cId = 0 if module == torch else 2
    if len(thing.shape) == 4 : 
        return cId + 1
    elif len(thing.shape) == 3 : 
        return cId
    else : 
        return None

def channels (thing, module=torch) : 
    idx = channelDim(thing, module)
    if idx is not None:  
        return thing.shape[idx]
    return None

def isImage (thing, module=torch) : 
    return (len(thing.shape) in [3, 4]) \
            and (channels(thing) in [1, 3, 4])

def isGreyScale (thing, module=torch) : 
    return isImage(thing, module) \
            and channels(thing) == 1

def toGreyScale (im, module=torch) : 
    return torch.cat((im, im, im), channelDim(im, module))

def moduleGradNorm (module) : 
    with torch.no_grad() : 
        paramsWithGrad = list(filter(
            lambda p: p.grad is not None, 
            module.parameters()
        ))
        nelts = list(map(lambda p: p.nelement(), paramsWithGrad))
        gradNorms = list(map(
            lambda p: p.pow(2).mean().item(), 
            paramsWithGrad
        ))
        numerator = sum(map(lambda x, y : x * y, nelts, gradNorms))
        denominator = sum(nelts) + 1e-3
        return numerator / denominator

def freezeLayers (model, freeze_layers) :
    for name, module in model.named_modules() : 
        if name in freeze_layers : 
            module.requires_grad_(False)

def convBackbone (opts) : 
    if opts.backbone == 'resnet18' : 
        model = resnet18(pretrained=True)
        inFeatures = model.fc.in_features
        model.fc = nn.Linear(inFeatures, opts.embedding_size)
        model.fc.apply(getInitializer(opts))
    elif opts.backbone == 'alexnet' : 
        model = alexnet(pretrained=True)
        inFeatures = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(inFeatures, opts.embedding_size)
        model.classifier[-1].apply(getInitializer(opts))
    else : 
        raise ValueError(f'{opts.backbone} not supported')
    freezeLayers(model, opts.freeze_layers)
    model = model.float()
    return model
