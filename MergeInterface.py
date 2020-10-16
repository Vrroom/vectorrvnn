import torch
from itertools import product
from matching import bestAssignmentCost
import os
from OneMergeAutoEncoder import OneMergeAutoEncoder
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

class MergeInterface (ttools.ModelInterface) : 

    def __init__(self, model, lr=1e-4, cuda=True, max_grad_norm=10,
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
        ])

    def _oneMergeCandidates (self, batch) : 
        allLeaf = lambda n, t : set(leaves(t)).issuperset(set(t.neighbors(n)))
        paths = []
        bboxes = []
        for t in batch : 
            for n in nonLeaves(t) : 
                if allLeaf(n, t) : 
                    paths.append(torch.stack([t._path(_) for _ in t.neighbors(n)]))
        paths = nn.utils.rnn.pad_sequence(paths, batch_first=True)
        ret = {}
        ret['paths'] = paths.to(self.device)
        return ret

    def _matchingcost (self, a, b, c) : 
        costs = [(c(c1.squeeze(), c2.squeeze())) for c1, c2 in product(a, b)]
        allpairs = product(range(len(a)), range(len(b)))
        costtable = dict(zip(allpairs, costs))
        return bestAssignmentCost(costtable)
        
    def forward(self, batch):
        paths = batch['paths']
        numNeighbors = batch['numNeighbors']
        paths = paths.to(self.device)
        out = self.model(paths, numNeighbors)
        return out

    def backward(self, batch, fwd_data):
        paths = batch['paths']
        numNeighbors = batch['numNeighbors']
        paths = paths.to(self.device)
        bs, *_ = paths.shape
        # Find min-cost matching sequentially
        loss = 0
        for path, recon, i in zip(paths, fwd_data, numNeighbors) : 
            path_ = path[:i]
            loss += self._matchingcost(path_, recon, F.mse_loss)
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
        return ret

    def _individualLosses (self, batch, fwd_data) : 
        paths = batch['paths']
        numNeighbors = batch['numNeighbors']
        paths = paths.to(self.device)
        bs, *_ = paths.shape
        # Find min-cost matching sequentially
        losses = []
        for path, recon, i in zip(paths, fwd_data, numNeighbors) : 
            path_ = path[:i]
            losses.append(float(self._matchingcost(path_, recon, F.mse_loss)))
        return losses

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


def train () : 
    def collate_fn (batch) : 
        allLeaf = lambda n, t : set(leaves(t)).issuperset(set(t.neighbors(n)))
        paths = []
        numNeighbors = []
        for t in batch : 
            for n in nonLeaves(t) : 
                if allLeaf(n, t) : 
                    paths.append(torch.stack([t._path(_) for _ in t.neighbors(n)]))
                    numNeighbors.append(t.out_degree(n))
        paths = nn.utils.rnn.pad_sequence(paths, batch_first=True)
        ret = dict(paths=paths, numNeighbors=numNeighbors)
        return ret

    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    # Load all the data
    trainDir = commonConfig['train_directory']
    trainData = SVGDataSet(trainDir, 'adjGraph', 10)
    trainData.toTensor()
    dataLoader = torch.utils.data.DataLoader(
        trainData, 
        batch_size=2, 
        shuffle=True,
        collate_fn=collate_fn
    )
    # Load pretrained path module
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VAE_OUTPUT = os.path.join(BASE_DIR, "results", "path_vae")
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", "merge")
    pathVAE = PathVAE(config)
    state_dict = torch.load(os.path.join(VAE_OUTPUT, 'training_end.pth'))
    pathVAE.load_state_dict(state_dict['model'])
    # Initiate main model.
    model = OneMergeAutoEncoder(pathVAE, config['sampler'])
    checkpointer = ttools.Checkpointer(MERGE_OUTPUT, model)
    interface = MergeInterface(model)
    trainer = ttools.Trainer(interface)
    port = 8097
    keys = ["loss", "wd"]
    name = "path_vae"
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
    train()
