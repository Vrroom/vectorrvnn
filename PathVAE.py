import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dictOps import aggregateDict
from Dataset import SVGDataSet
from PathModules import * 
import torch
from torch import nn
import torch.optim as optim
from VAEInterface import VAEInterface
import ttools 
import ttools.interfaces
from ttools.modules import networks
import visdom
from losses import iou

LOG = ttools.get_logger(__name__)

class BBoxInterface (ttools.ModelInterface) :
    def __init__ (self, featureModel, model, lr=1e-4, cuda=True, max_grad_norm=10):
        super(BBoxInterface, self).__init__()
        self.max_grad_norm = max_grad_norm
        self.featureModel = featureModel
        self.model = model
        self.device = "cpu"
        if cuda : 
            self.device = "cuda"
        self.model.to(self.device)
        self.featureModel.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
    
    def forward (self, batch): 
        paths, _ = batch
        paths = paths.to(self.device)
        z, _ = self.featureModel.encode(paths)
        return self.model(z)

    def backward(self, batch, fwd_data) : 
        ret = {}
        prediction = fwd_data
        _, bboxes = batch
        bboxes = bboxes.to(self.device)
        iouLoss = iou(prediction, bboxes)
        iouLoss = -iouLoss.mean()
        ret['iou'] = iouLoss.item()
        reg_loss = 0
        for p in self.model.parameters() : 
            reg_loss += p.pow(2).sum()
        ret['wd'] = reg_loss.item()
        self.opt.zero_grad()
        iouLoss.backward()
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()
        return ret

class EpochUpdateCallback(ttools.callbacks.Callback) : 
    def __init__ (self, interface) : 
        super(EpochUpdateCallback, self).__init__()
        self.interface = interface

    def epoch_end(self) : 
        super(EpochUpdateCallback, self).epoch_end()
        self.interface.epoch += 1

class BBoxVisCallback(ttools.callbacks.Callback):
    def __init__(self, frequency=100, server=None, port=8097, base_url="/", env="main", win="BBoxVis"):
        super(BBoxVisCallback, self).__init__()
        if server is None:
            server = "http://localhost"
        self._api = visdom.Visdom(server=server, port=port, env=env,
                                  base_url=base_url)

        self.opts = {
            'layoutopts': {
                'plotly': {
                    'xaxis': {
                        'range': [0, 1],
                        'autorange': False,
                    },
                    'yaxis': {
                        'range': [0, 1],
                        'autorange': False,
                    }
                }
            }
        }
        self._api.line(torch.zeros(1), name="ogBox", win=win, opts=self.opts)
        self._api.line(torch.zeros(1), name="predictedBox", win=win, opts=self.opts)
        self.win = win
        self._step = 0
        self.frequency = frequency

    def _box2lines (self, box, name, opts) : 
        x = torch.stack([box[0], box[0], box[0] + box[2], box[0] + box[2], box[0]])
        y = torch.stack([box[1], box[1] + box[3], box[1] + box[3], box[1], box[1]])
        self._api.line(x, y, self.win, update="replace", name=name, opts=opts)

    def batch_end(self, batch_data, train_step_data, bwd_result):
        super(BBoxVisCallback, self).batch_end(batch_data, train_step_data, bwd_result)
        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0
        ogBox = batch_data[1][0].cpu()
        predictedBox = train_step_data[0].cpu().detach()
        opts_ = {'linecolor':np.array([[255, 0, 0]]), **self.opts}
        self._box2lines(ogBox, "ogBox", self.opts)
        self._box2lines(predictedBox, "predictedBox", opts_)
        self._step += 1

    def validation_end(self, val_data):
        super(BBoxVisCallback, self).validation_end(val_data)

class PathVisCallback(ttools.callbacks.Callback):
    def __init__(self, frequency=100, server=None, port=8097, base_url="/", env="main", win="PathVis"):
        super(PathVisCallback, self).__init__()
        if server is None:
            server = "http://localhost"
        self._api = visdom.Visdom(server=server, port=port, env=env,
                                  base_url=base_url)

        self.opts = {
            'layoutopts': {
                'plotly': {
                    'xaxis': {
                        'range': [0, 1],
                        'autorange': False,
                    },
                    'yaxis': {
                        'range': [0, 1],
                        'autorange': False,
                    }
                }
            }
        }
        self._api.line(torch.zeros(1), name="path", win=win, opts=self.opts)
        self._api.line(torch.zeros(1), name="reconPath", win=win, opts=self.opts)
        self.win = win
        self._step = 0
        self.frequency = frequency

    def batch_end(self, batch_data, train_step_data, bwd_result):
        super(PathVisCallback, self).batch_end(batch_data, train_step_data, bwd_result)

        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0

        paths = batch_data['paths'].cpu()
        path = paths[0][0]
        reconPath = train_step_data[0][0].cpu().detach()
        opts_ = {'linecolor':np.array([[255, 0, 0]]), **self.opts}
        self._api.line(path[0], path[1], win=self.win, update="replace", name="path", opts=self.opts)
        self._api.line(reconPath[0], reconPath[1], win=self.win, update="replace", name="reconPath", opts=opts_)

        self._step += 1

    def validation_end(self, val_data):
        super(PathVisCallback, self).validation_end(val_data)

class PathVAE (nn.Module) : 

    def __init__ (self, config) : 
        super(PathVAE, self).__init__() 
        input_size = config['sampler']['input_size']
        hidden_size = config['sampler']['hidden_size']
        self.nnMu = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.SELU(),
            nn.Linear(hidden_size, input_size)
        )
        self.nnLogVar = nn.Sequential(
            nn.Linear(input_size, hidden_size), 
            nn.SELU(),
            nn.Linear(hidden_size, input_size)
        )
        self.pathEncoder = MLPPathEncoder(config['pathEncoder'])
        self.pathDecoder = MLPPathDecoder(config['pathDecoder'])
        self.lossWeights = config['lossWeights']
        self.mseLoss = nn.MSELoss()

    def sample (self) : 
        eps = torch.randn(1, self.sampler.input_size)
        pts = self.pathDecoder(eps).detach().numpy()
        return pts.squeeze().T

    def reconstruct (self, x) : 
        e = self.pathEncoder(x)
        root, _ = torch.chunk(self.sampler(e), 2, 1)
        x_ = self.pathDecoder(root)
        return x_

    def encode(self, x):
        out = self.pathEncoder(x)
        mu = self.nnMu(out)
        logvar = self.nnLogVar(out)
        return mu, logvar

    def decode(self, z):
        x_ = self.pathDecoder(z)
        aux_data = {}
        return x_, aux_data

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(logvar)
        return mu + std*eps

    def forward (self, x) : 
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_, aux =  self.decode(z)
        aux["logvar"] = logvar
        aux["mu"] = mu
        return x_, aux

def vaeTrain() :
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    with open('./Configs/config.json') as fd: 
        config = json.load(fd)
    # Get all the directory names
    trainDir = commonConfig['train_directory']
    # Load all the data
    trainData = SVGDataSet(trainDir, 'adjGraph', 10)
    trainData.toTensor()
    num_epochs = 1000
    descriptors = torch.cat([t.descriptors for t in trainData])
    dataLoader = torch.utils.data.DataLoader(
        descriptors, 
        batch_size=config['batch_size'], 
        shuffle=True,
    )
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VAE_OUTPUT = os.path.join(BASE_DIR, "results", "path_vae")
    model = PathVAE(config)
    checkpointer = ttools.Checkpointer(VAE_OUTPUT, model)
    extras, meta = checkpointer.load_latest()
    interface = VAEInterface(model, num_epochs, variational=True)
    trainer = ttools.Trainer(interface)
    port = 8097
    keys = ["kld", "data_loss", "loss", "wd"]
    name = "path_vae"
    trainer.add_callback(EpochUpdateCallback(interface))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=keys, val_keys=keys))
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=keys, env=name, port=port))
    trainer.add_callback(PathVisCallback(
        env=name, win="samples", port=port, frequency=1000))
    # Start training
    trainer.train(dataLoader, num_epochs=num_epochs)

def bboxTrain () : 
    from Model import bboxModel
    def collate_fn (batch) : 
        descriptors = torch.cat([t.descriptors for t in batch])
        bbox = torch.cat([t.pathViewBoxes for t in batch])
        return descriptors, bbox
    with open('commonConfig.json') as fd : 
        commonConfig = json.load(fd)
    with open('./Configs/config.json') as fd: 
        config = json.load(fd)
    trainDir = commonConfig['train_directory']
    # Load all the data
    trainData = SVGDataSet(trainDir, 'adjGraph', 10)
    trainData.toTensor()
    num_epochs = 1000
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    VAE_OUTPUT = os.path.join(BASE_DIR, "results", "path_vae")
    dataLoader = torch.utils.data.DataLoader(
        trainData, 
        batch_size=config['batch_size'], 
        shuffle=True,
        collate_fn=collate_fn
    )
    featureModel = PathVAE(config)
    state_dict = torch.load(os.path.join(VAE_OUTPUT, 'training_end.pth'))
    featureModel.load_state_dict(state_dict['model'])
    model = bboxModel(config['sampler'])
    interface = BBoxInterface(featureModel, model)
    trainer = ttools.Trainer(interface)
    port = 8097
    keys = ["iou", "wd"]
    name = "path_vae"
    trainer.add_callback(BBoxVisCallback(env=name, port=port, frequency=1000))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=keys, val_keys=keys))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=keys, env=name, port=port))
    # Start training
    trainer.train(dataLoader, num_epochs=num_epochs)
    
if __name__ == "__main__" :
    bboxTrain()
