import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import random
from torchvision import transforms as T
from listOps import avg
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from itertools import product
import os
from TripletNet import *
from Dataset import SVGDataSet
import json
import more_itertools
import numpy as np
import math
import ttools 
import ttools.interfaces
from ttools.modules import networks
from dictOps import aggregateDict
import visdom

LOG = ttools.get_logger(__name__)

class KernelCallback (ttools.callbacks.ImageDisplayCallback) : 
    def __init__ (self, key, env, win, port, frequency=100) : 
        super(KernelCallback, self).__init__(env=env, win=win, port=port, frequency=frequency)
        self.key = key

    def visualized_image(self, batch, step_data, is_val):
        if not is_val : 
            kernels = step_data[self.key].cpu()
            N, *_ = kernels.shape
            chunks = torch.chunk(kernels, N)
            chunks = [c.squeeze() for c in chunks]
            chunks = [(c - c.min()) / (c.max() - c.min()) for c in chunks]
            n = math.isqrt(N)
            viz =  torch.stack([torch.cat(chunks[i*n: (i+1)*n], 1) for i in range(n)])
            return viz

    def caption(self, batch, step_data, is_val):
        if not is_val: 
            return self.key

# TODO SHOW ALL 4 IMAGES
class ImageCallback(ttools.callbacks.ImageDisplayCallback):
    def visualized_image(self, batch, step_data, is_val):
        if is_val : 
            N = batch['im'].shape[0]
            i = random.randint(0, N-1)
            im = batch['im'][i:i+3].cpu()
            im1 = batch['im1'][i:i+3].cpu()
            im2 = batch['im2'][i:i+3].cpu()
            viz = torch.cat([im, im1, im2], 2)
            viz = (viz - viz.min()) / (viz.max() - viz.min())
            return viz
        else : 
            im = batch['im'][0].cpu()
            im1 = batch['im1'][0].cpu()
            im2 = batch['im2'][0].cpu()
            viz = torch.cat([im, im1, im2], 2)
            viz = (viz - viz.min()) / (viz.max() - viz.min())
            return viz
        
    def caption(self, batch, step_data, is_val):
        # write some informative caption into the visdom window
        if not is_val : 
            ref = float(batch['lca'][0])
            pred = float(step_data['pred']['score'][0])
            s = 'REF: {:10.4f}, PRED: {:10.4f}'.format(ref, pred)
            return s
        return ''

class TripletInterface (ttools.ModelInterface) : 

    def __init__(self, model, lr=3e-4, cuda=True, max_grad_norm=10,
                 variational=True):
        super(TripletInterface, self).__init__()
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.device = "cpu"
        self.epoch = 0
        self.cuda = cuda
        if cuda:
            self.device = "cuda"
        self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def logActivations (self, ret) : 
        for name, module in self.model.named_children() : 
            relus = filter(lambda x : isinstance(x, RecordingReLU), module.modules())
            activated = list(map(lambda x : x.activated, relus))
            ret[f'{name}_activated'] = avg(activated)

    def logOutNorms (self, ret) : 
        for name, module in self.model.named_children() : 
            ret[f'{name}_outnorm'] = module.outNorm

    def logGradients (self, ret) : 
        with torch.no_grad() : 
            for name, module in self.model.named_children() : 
                grad = torch.tensor(0.).cuda()
                for p in module.parameters() : 
                    if p.grad is not None : 
                        grad += p.grad.pow(2).mean()
                grad /= len(list(module.parameters())) + 1
                ret[f'{name}_grad'] = grad.item()

    def logParameterNorms (self, ret) : 
        with torch.no_grad() :
            for name, module in self.model.named_children() : 
                wd = torch.tensor(0.).cuda()
                for p in module.parameters() : 
                    wd += p.pow(2).mean()
                wd /= len(list(module.parameters())) + 1
                ret[f'{name}_wd'] = wd.item()

    def training_step(self, batch) :
        im = batch['im'].cuda()
        ref = batch['ref'].cuda()
        plus = batch['plus'].cuda()
        minus = batch['minus'].cuda()
        dplus2 = self.model(im, ref, plus, minus)
        loss = dplus2.sum()
        ret = {}
        # optimize
        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()
        ret["loss"] = loss.item()
        ret["last-layer-bias"] = self.model.nn2[-2].bias
        ret["conv-first-layer-kernel"] = self.model.conv.module[0].weight
        self.logParameterNorms(ret)
        self.logGradients(ret)
        self.logActivations(ret)
        self.logOutNorms(ret)
        return ret

    def init_validation(self):
        return {"count": 0, "loss": 0}

    def validation_step(self, batch, running_data) : 
        with torch.no_grad():
            im = batch['im'].cuda()
            ref = batch['ref'].cuda()
            plus = batch['plus'].cuda()
            minus = batch['minus'].cuda()
            dplus2 = self.model(im, ref, plus, minus)
            loss = dplus2.sum().item()
            n = lca.numel()
            count = running_data['count']
            cumLoss = (running_data["loss"] * count + loss * n) / (count + n)
        return {
            "loss" : cumLoss,
            "count": running_data["count"] + n, 
        }


def train (name) : 
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    trainData = SVGDataSet('train.pkl')
    dataLoader = torch.utils.data.DataLoader(
        trainData, 
        batch_size=128, 
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda x : aggregateDict(x, torch.stack)
    )
    valData = SVGDataSet('cv.pkl')
    valDataLoader = torch.utils.data.DataLoader(
        valData, 
        batch_size=128,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda x : aggregateDict(x, torch.stack)
    )
    # Load pretrained path module
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", name)
    # Initiate main model.
    model = RasterLCAModel(config['sampler']).float()
    checkpointer = ttools.Checkpointer(MERGE_OUTPUT, model)
    interface = TripletInterface(model)
    trainer = ttools.Trainer(interface)
    port = 8097
    named_children = [n for n, _ in model.named_children()]
    named_grad = [f'{n}_grad' for n in named_children]
    named_wd = [f'{n}_wd' for n in named_children]
    named_activated = [f'{n}_activated' for n in named_children]
    named_outnorms = [f'{n}_outnorm' for n in named_children]
    keys = ["loss", *named_grad, *named_wd, *named_activated, *named_outnorms]
    val_keys=keys[:1]
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=val_keys, val_keys=val_keys))
    trainer.add_callback(ImageCallback(env=name + "_vis", win="samples", port=port, frequency=100))
    trainer.add_callback(KernelCallback(key="conv-first-layer-kernel", env=name + "_kernel", win="conv", port=port))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=val_keys, env=name + "_training_plots", port=port, frequency=100))
    # Start training
    trainer.train(dataLoader, num_epochs=4000, val_dataloader=valDataLoader)

if __name__ == "__main__" : 
    import sys
    train(sys.argv[1])

