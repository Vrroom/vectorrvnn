import torch
from itertools import product
from matching import bestAssignmentCost
import os
from RasterLCAModel import RasterLCAModel
from PathVAE import PathVAE, PathVisCallback
from Dataset import SVGDataSet
import json
import more_itertools
import numpy as np
import math
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import ttools 
import ttools.interfaces
from ttools.modules import networks
from treeOps import leaves, nonLeaves
from dictOps import aggregateDict

LOG = ttools.get_logger(__name__)

class ImageCallback(ttools.callbacks.ImageDisplayCallback):
    """Simple callback that visualize images."""
    def visualized_image(self, batch, fwd_result):
        ref1 = batch[0][0].cpu()
        ref2 = batch[1][0].cpu()
        viz = torch.clamp(torch.cat([ref1, ref2], 2), 0, 1)
        return viz
    
    def caption(self, batch, fwd_result):
        # write some informative caption into the visdom window
        ref = float(batch[2][0][0])
        pred = float(fwd_result[0][0])
        s = 'REF: {:10.4f}, PRED: {:10.4f}'.format(ref, pred)
        return s

class MergeInterface (ttools.ModelInterface) : 

    def __init__(self, model, lr=5e-4, cuda=True, max_grad_norm=10,
                 variational=True):
        super(MergeInterface, self).__init__()
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.device = "cpu"
        self.epoch = 0
        self.cuda = cuda
        if cuda:
            self.device = "cuda"
        self.model.to(self.device)
        self.loss = nn.L1Loss()
        self.opt = optim.Adam([
            {'params': self.model.resnet18.parameters(), 'lr': 1e-7},
            {'params': self.model.nn1.parameters(), 'lr': lr},
            {'params': self.model.nn2.parameters(), 'lr': lr}
        ])

    def forward(self, batch):
        return self.model(batch[0].cuda(), batch[1].cuda())

    def backward(self, batch, fwd_data):
        loss = self.loss(fwd_data, batch[2].cuda())
        ret = {}
        reg_loss = 0
        for p in self.model.parameters():
            reg_loss += p.pow(2).sum()
        ret["wd"] = reg_loss.item()
        # optimize
        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()
        ret["loss"] = loss.item()
        return ret

    def init_validation(self):
        return {"count": 0, "loss": 0}

    def update_validation(self, batch, fwd, running_data):
        with torch.no_grad():
            loss = self.loss(fwd, batch[2].cuda())
            n = batch[2].numel()
        return {
            "loss": running_data["loss"] + loss.item()*n,
            "count": running_data["count"] + n
        }

    def finalize_validation(self, running_data):
        reg_loss = 0
        for p in self.model.parameters():
            reg_loss += p.pow(2).sum()
        reg_loss = reg_loss.item()
        return {
            "loss": running_data["loss"] / running_data["count"],
            "wd": reg_loss
        }


def train (name) : 
    def collate_fn (batch) : 
        im1 = torch.stack([b[0] for b in batch])
        im2 = torch.stack([b[1] for b in batch])
        y = torch.stack([b[2] for b in batch])
        return im1, im2, y

    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    # Load all the data
    trainData = SVGDataSet('__train__.h5')
    dataLoader = torch.utils.data.DataLoader(
        trainData, 
        batch_size=128, 
        shuffle=True,
        collate_fn=collate_fn
    )
    valData = SVGDataSet('__cv__.h5')
    valDataLoader = torch.utils.data.DataLoader(
        valData, 
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )
    # Load pretrained path module
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", name)
    # Initiate main model.
    model = RasterLCAModel(config['sampler']).float()
    checkpointer = ttools.Checkpointer(MERGE_OUTPUT, model)
    interface = MergeInterface(model)
    trainer = ttools.Trainer(interface)
    port = 8097
    keys = ["loss", "wd"]
    trainer.add_callback(ImageCallback(
        env=name, win="samples", port=port, frequency=10))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=keys, val_keys=keys))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=keys, env=name, port=port, frequency=10))
    # Start training
    trainer.train(dataLoader, num_epochs=1000, val_dataloader=valDataLoader)

if __name__ == "__main__" : 
    import sys
    train(sys.argv[1])
