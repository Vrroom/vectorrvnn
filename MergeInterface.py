import torch
import random
from torchvision import transforms as T
from listOps import avg
from losses import iou
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
from dictOps import aggregateDict
from itertools import product
from matching import bestAssignmentCost
import os
from RasterLCAModel import *
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
import visdom

LOG = ttools.get_logger(__name__)

class LastLayerBiasPlotCallback (ttools.callbacks.Callback) : 
    def __init__(self, frequency=100, server=None, port=8097, base_url="/", env="main", log=False):
        super(LastLayerBiasPlotCallback, self).__init__()
        if server is None:
            server = "http://localhost"
        self._api = visdom.Visdom(server=server, port=port, env=env,
                                  base_url=base_url)
        if self._api.win_exists("last-layer-biases"):
            self._api.close("last-layer-biases")
        self._step = 0
        self.frequency = frequency

    def batch_end(self, batch, step_data) :
        super(LastLayerBiasPlotCallback, self).batch_end(batch, step_data)

        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0

        t = self.batch / max(self.datasize, 1) + self.epoch
        bias = step_data['last-layer-bias'].cpu().detach().numpy()
        opts = {
            "xlabel": "epoch", 
            "ylabel": "biases",
            "legend": [f'bias-{i}' for i in range(len(bias))]
        }
        for i, b in enumerate(bias) : 
            self._api.line([b], [t], update="append", win="last-layer-biases", name=f'bias-{i}', opts=opts)

        self._step += 1

class ConfusionLineCallback(ttools.callbacks.Callback):
    def __init__(self, frequency=100, server=None, port=8097, base_url="/", env="main", log=False):
        super(ConfusionLineCallback, self).__init__()
        if server is None:
            server = "http://localhost"
        self._api = visdom.Visdom(server=server, port=port, env=env,
                                  base_url=base_url)

        if self._api.win_exists("confusion"):
            self._api.close("confusion")
        if self._api.win_exists("val_confusion"):
            self._api.close("val_confusion")
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
        self._step = 0
        self.frequency = frequency

    def epoch_start (self, epoch) : 
        super(ConfusionLineCallback, self).epoch_start(epoch)
        self.lca = None
        self.predLca = None 

    def plot_data (self, lca, predLca, win) : 
        ps = np.arange(0.1, 1.01, 0.1)
        xs = []
        ys = []
        ss = []
        for p in ps : 
            ind = np.abs(lca - p) < 1e-3
            if ind.any(): 
                arr = predLca[ind]
                xs.append(p)
                ys.append(arr.mean())
                ss.append(arr.std())
        xs = np.array(xs)
        ys = np.array(ys)
        ss = np.array(ss)
        fig, ax = plt.subplots()
        ax.plot(xs, ys) 
        ax.fill_between(xs, ys - ss, ys + ss, alpha=0.1)
        ax.set_xlabel("depth difference / 10")
        ax.set_ylabel("prediction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.suptitle(win)
        self._api.matplot(plt, win=win)
        plt.close(fig)

    def batch_end(self, batch, step_data) :
        super(ConfusionLineCallback, self).batch_end(batch, step_data)
        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0
        lca = batch['lca'].cpu().detach().numpy().flatten()
        predLca = step_data['pred']['score'].cpu().detach().numpy()
        self.lca = lca if self.lca is None else np.concatenate([lca, self.lca])
        self.predLca = predLca if self.predLca is None else np.concatenate([predLca, self.predLca])
        self.plot_data(self.lca, self.predLca, "confusion")
        self._step += 1

    def val_batch_end (self, batch_data, running_val_data) : 
        super(ConfusionLineCallback, self).val_batch_end(batch_data, running_val_data)
        lca = running_val_data['lca'].cpu().detach().numpy().flatten()
        predLca = running_val_data['predLca'].cpu().detach().numpy()
        self.plot_data(lca, predLca, "val_confusion") 
        self._step += 1


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
                        'range': [-1, 0],
                        'autorange': False,
                    }
                }
            }
        }
        self._api.line(torch.zeros(1), name="box1", win=win, opts=self.opts)
        self._api.line(torch.zeros(1), name="box2", win=win, opts=self.opts)
        self.win = win
        self._step = 0
        self.frequency = frequency

    def _box2lines (self, box, name, opts) : 
        x = torch.stack([box[0], box[0], box[0] + box[2], box[0] + box[2], box[0]])
        box = -box
        y = torch.stack([box[1], box[1] + box[3], box[1] + box[3], box[1], box[1]])
        self._api.line(y, x, self.win, update="replace", name=name, opts=opts)

    def batch_end(self, batch, step_data):
        super(BBoxVisCallback, self).batch_end(batch, step_data)
        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0
        box1 = batch['bbox1'][0].cpu()
        box2 = batch['bbox2'][0].cpu()
        box1Pred = step_data['pred']['box1Pred'][0].cpu()
        box2Pred = step_data['pred']['box2Pred'][0].cpu()
        opts_ = {'linecolor':np.array([[255, 0, 0]]), **self.opts}
        self._box2lines(box1, "box1", self.opts)
        self._box2lines(box2, "box2", opts_)
        self._box2lines(box1Pred, "box1Pred", self.opts)
        self._box2lines(box2Pred, "box2Pred", opts_)
        self._step += 1

class ImageCallback(ttools.callbacks.ImageDisplayCallback):
    def visualized_image(self, batch, step_data, is_val):
        im = batch['im'][0].cpu()
        im1 = batch['im1'][0].cpu()
        im2 = batch['im2'][0].cpu()
        viz = torch.cat([im, im1, im2], 2)
        viz = (viz - viz.min()) / (viz.max() - viz.min())
        return viz
        
    def caption(self, batch, step_data, is_val):
        # write some informative caption into the visdom window
        if is_val : 
            ref = float(step_data['lca'][0])
            pred = float(step_data['predLca'][0])
        else : 
            ref = float(batch['lca'][0])
            pred = float(step_data['pred']['score'][0])
        s = 'REF: {:10.4f}, PRED: {:10.4f}'.format(ref, pred)
        return s

class MergeInterface (ttools.ModelInterface) : 

    def __init__(self, model, lr=3e-4, cuda=True, max_grad_norm=10,
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
        self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        # self.opt = optim.Adam([
        #     {'params': self.model.alexnet1.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        #     {'params': self.model.alexnet2.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        #     {'params': self.model.boxEncoder.parameters(), 'lr': lr},
        #     {'params': self.model.nn1.parameters(), 'lr': lr, 'weight_decay': weight_decay},
        #     {'params': self.model.nn2.parameters(), 'lr': lr, 'weight_decay':weight_decay}
        # ])

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
        im1 = batch['im1'].cuda()
        im2 = batch['im2'].cuda()
        bbox1 = batch['bbox1'].cuda()
        bbox2 = batch['bbox2'].cuda()
        fwd_data = self.model(im, im1, im2, bbox1, bbox2)
        lca = batch['lca'].cuda()
        scoreLoss = self.loss(fwd_data['score'], lca)
        iouLoss = (-iou(bbox1, fwd_data['box1Pred']).mean() - iou(bbox2, fwd_data['box2Pred']).mean()) / 2
        totalLoss = scoreLoss + iouLoss
        ret = {}
        # optimize
        self.opt.zero_grad()
        totalLoss.backward()
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()
        ret["scoreLoss"] = scoreLoss.item()
        ret["iouLoss"] = iouLoss.item()
        ret["totalLoss"] = totalLoss.item()
        ret["pred"] = fwd_data
        ret["last-layer-bias"] = self.model.nn2[-2].bias
        ret["alexnet1-first-layer-kernel"] = self.model.alexnet1.module[0].weight
        # ret["alexnet2-first-layer-kernel"] = self.model.alexnet2.module[0].weight
        self.logParameterNorms(ret)
        self.logGradients(ret)
        self.logActivations(ret)
        self.logOutNorms(ret)
        return ret

    def init_validation(self):
        return {"count": 0, "scoreLoss": 0, "iouLoss": 0, "totalLoss": 0, "lca": None, "predLca": None}

    def validation_step(self, batch, running_data) : 
        with torch.no_grad():
            lca = batch['lca'].cuda()
            im = batch['im'].cuda()
            im1 = batch['im1'].cuda()
            im2 = batch['im2'].cuda()
            bbox1 = batch['bbox1'].cuda()
            bbox2 = batch['bbox2'].cuda()
            fwd = self.model(im, im1, im2, bbox1, bbox2)
            scoreLoss = self.loss(fwd['score'], lca)
            iouLoss = (-iou(bbox1, fwd['box1Pred']).mean() - iou(bbox2, fwd['box2Pred']).mean()) / 2
            totalLoss = scoreLoss + iouLoss
            n = lca.numel()
            lcaNew = batch['lca'] if running_data['lca'] is None else torch.cat([batch['lca'], running_data['lca']])
            predLcaNew = fwd['score'] if running_data['predLca'] is None else torch.cat([fwd['score'], running_data['predLca']])
            cumScoreLoss = (running_data["scoreLoss"] * running_data["count"] + scoreLoss.item() * n) / (running_data["count"] + n)
            cumIouLoss = (running_data["iouLoss"] * running_data["count"] + iouLoss.item() * n) / (running_data["count"] + n)
            cumTotalLoss = (running_data["totalLoss"] * running_data["count"] + totalLoss.item() * n) / (running_data["count"] + n)
        return {
            "scoreLoss" : cumScoreLoss,
            "iouLoss":  cumIouLoss,
            "totalLoss": cumTotalLoss,
            "count": running_data["count"] + n, 
            "lca": lcaNew,
            "predLca": predLcaNew
        }


def train (name) : 
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    # Load all the data
    # transform = T.ColorJitter(0.1,0.1,0.1,0.1)
    trainData = SVGDataSet('train.pkl') #, transform)
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
    interface = MergeInterface(model)
    trainer = ttools.Trainer(interface)
    port = 8097
    named_children = [n for n, _ in model.named_children()]
    named_grad = [f'{n}_grad' for n in named_children]
    named_wd = [f'{n}_wd' for n in named_children]
    named_activated = [f'{n}_activated' for n in named_children]
    named_outnorms = [f'{n}_outnorm' for n in named_children]
    keys = ["scoreLoss", "iouLoss", "totalLoss", *named_grad, *named_wd, *named_activated, *named_outnorms]
    val_keys=keys[:3]
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(
        keys=val_keys, val_keys=val_keys))
    trainer.add_callback(ConfusionLineCallback(env=name + "_confusion", port=port, frequency=100))
    trainer.add_callback(ImageCallback(env=name + "_vis", win="samples", port=port, frequency=50))
    trainer.add_callback(BBoxVisCallback(env=name + "_vis", port=port, frequency=50))
    trainer.add_callback(LastLayerBiasPlotCallback(env=name + "_bias", port=port, frequency=100))
    trainer.add_callback(KernelCallback(key="alexnet1-first-layer-kernel", env=name + "_kernel", win="alexnet1", port=port))
    # trainer.add_callback(KernelCallback(key="alexnet2-first-layer-kernel", env=name + "_kernel", win="alexnet2", port=port))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=val_keys, env=name + "_training_plots", port=port, frequency=100))
    # Start training
    trainer.train(dataLoader, num_epochs=4000, val_dataloader=valDataLoader)

if __name__ == "__main__" : 
    import sys
    train(sys.argv[1])
