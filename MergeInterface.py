import torch
from itertools import product
from matching import bestAssignmentCost
import os
from Model import VectorRvNNAutoEncoder
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

class MergeInterface (ttools.ModelInterface) : 

    def __init__(self, model, lr=1e-4, cuda=False, max_grad_norm=10,
                 variational=True):
        super(MergeInterface, self).__init__()
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.device = "cpu"
        self.epoch = 0
        if cuda:
            self.device = "cuda"
        self.model.to(self.device)
        self.opt = optim.Adam([
            {'params': self.model.pathVAE.parameters(), 'lr': 1e-7},
            {'params': self.model.mergeEncoder.parameters(), 'lr': lr},
            {'params': self.model.mergeDecoder.parameters(), 'lr': lr},
            {'params': self.model.splitter.parameters(), 'lr': lr}, 
            {'params': self.model.bbox.parameters(), 'lr': lr}
        ])

    def forward(self, batch):
        for t in batch: 
            t.descriptors.to(self.device)
        return [self.model._forward(b) for b in batch]

    def backward(self, batch, fwd_data):
        for t in batch: 
            t.descriptors.to(self.device)
        losses = [self.model._backward(t, p, b) for (t, (p, b)) in zip(batch, fwd_data)]
        losses = aggregateDict(losses, sum)
        descReconLoss = losses['descReconLoss']
        bboxLoss = losses['bboxLoss']
        loss  = descReconLoss + bboxLoss
        bs = len(batch)
        loss = loss / bs
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
        ret["bboxLoss"] = bboxLoss.item()
        ret["descReconLoss"] = descReconLoss.item()
        return ret

    def init_validation(self):
        return {"count": 0, "loss": 0}

    def update_validation(self, batch, fwd, running_data):
        with torch.no_grad():
            ref = batch[1].to(self.device)
            loss = F.mse_loss(fwd, ref)
            n = ref.shape[0]
        return {
            "loss": running_data["loss"] + loss.item()*n,
            "count": running_data["count"] + n
        }

    def finalize_validation(self, running_data):
        return {
            "loss": running_data["loss"] / running_data["count"]
        }


def train (path_vae_name, name) : 
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load all the data
    trainDir = commonConfig['train_directory']
    trainData = SVGDataSet(trainDir, 'adjGraph', 10, useColor=False)
    trainData.toTensor()
    dataLoader = torch.utils.data.DataLoader(
        trainData, 
        batch_size=16, 
        shuffle=True,
        collate_fn=lambda x : x
    )
    # Load pretrained path module
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VAE_OUTPUT = os.path.join(BASE_DIR, "results", path_vae_name)
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", name)
    pathVAE = PathVAE(config)
    state_dict = torch.load(os.path.join(VAE_OUTPUT, 'training_end.pth'), map_location=torch.device('cpu'))
    pathVAE.load_state_dict(state_dict['model'])
    # Initiate main model.
    model = VectorRvNNAutoEncoder(pathVAE, config)
    checkpointer = ttools.Checkpointer(MERGE_OUTPUT, model)
    interface = MergeInterface(model)
    trainer = ttools.Trainer(interface)
    port = 8097
    keys = ["loss", "wd", "bboxLoss", "descReconLoss"]
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    trainer.add_callback(PathVisCallback(
        env=name, win="samples", port=port, frequency=200))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=keys, val_keys=keys))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=keys, env=name, port=port))
    # Start training
    trainer.train(dataLoader, num_epochs=1000)

if __name__ == "__main__" : 
    import sys
    train(sys.argv[1], sys.argv[2])
