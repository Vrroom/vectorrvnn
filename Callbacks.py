import torch
from treeOps import *
import random
import numpy as np
import math
import ttools 
import ttools.interfaces
from ttools.modules import networks
import visdom
from functools import lru_cache
from listOps import avg
from treeCompare import *

class FMIndexCallback(ttools.callbacks.Callback):  

    def __init__(self, model, val_dataset, 
            frequency=100, server=None, port=8097, 
            base_url="/", env="main", log=False):
        if server is None:
            server = "http://localhost"
        self._api = visdom.Visdom(server=server, port=port, env=env,
                                  base_url=base_url)
        if self._api.win_exists("val_FM"):
            self._api.close("val_FM")
        self._step = 0
        self.model = model
        self.val_dataset = val_dataset
        self.frequency = frequency

    def scoreTree (self, t, gt) : 
        a = []
        for i in range(100) : 
            arrays = [hierarchicalClusterCompareFM(t1, t2, 6) for t1, t2 in zip(t, gt)]
            arrays = np.stack(arrays)
            a.append(np.mean(arrays, axis=0))
        a = np.stack(a)
        return torch.from_numpy(a.mean(axis=0))

    def validation_start(self, dataloader) : 
        super(FMIndexCallback, self).validation_start(dataloader)
        trees = random.sample([t for t in self.val_dataset.svgDatas if 20 > t.nPaths > 5], k=5)
        trees = [treeify(t) for t in trees]
        dendrograms = [self.model.dendrogram(t) for t in trees]
        greedyTrees = [self.model.greedyTree(t) for t in trees]
        greedyBinaryTrees = [self.model.greedyBinaryTree(t) for t in trees]
        dscores = self.scoreTree(dendrograms, trees)
        gtscores = self.scoreTree(greedyTrees, trees)
        gbtscores = self.scoreTree(greedyBinaryTrees, trees)
        x = torch.tensor(list(range(2, 6)))
        legend = ["dendrogram", "greedyTree", "greedyBinaryTree"]
        opts=dict(
            legend=legend, 
            title="FMIndexComparison", 
            xlabel="k",
            ylabel="FMIndex"
        )
        self._api.line(dscores, x, win="val_FM", name="dendrogram", opts=opts)
        self._api.line(gtscores, x, win="val_FM", update="append", name="greedyTree", opts=opts)
        self._api.line(gbtscores, x, win="val_FM", update="append", name="greedyBinaryTree", opts=opts)


class ConfusionDistanceCallback(ttools.callbacks.Callback):
    def __init__(self, model, dataset, val_dataset, 
            frequency=100, server=None, port=8097, 
            base_url="/", env="main", log=False):
        super(ConfusionDistanceCallback, self).__init__()
        if server is None:
            server = "http://localhost"
        self._api = visdom.Visdom(server=server, port=port, env=env,
                                  base_url=base_url)

        if self._api.win_exists("confusion_distance"):
            self._api.close("confusion_distance")
        if self._api.win_exists("val_confusion_distance"):
            self._api.close("val_confusion_distance")
        self._step = 0
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.frequency = frequency

    def epoch_start (self, epoch) : 
        super(ConfusionDistanceCallback, self).epoch_start(epoch)
        self.lca = None
        self.dlca = None 

    def getNodeEmbedding (self, tId, node, dataset) : 
        pt = dataset.getNodeInput(tId, node)
        im    = pt['im'].cuda()
        crop  = pt['crop'].cuda()
        whole = pt['whole'].cuda()
        return self.model.embedding(im, crop, whole)

    def sample (self, dataset, k=500) : 
        lcaScores = []
        dScores = []
        for i in range(k) : 
            tId = random.randint(0, len(dataset.svgDatas) - 1)
            t = dataset.svgDatas[tId]
            a, b = random.sample(t.nodes, k=2)
            ea = self.getNodeEmbedding(tId, a, dataset)
            eb = self.getNodeEmbedding(tId, b, dataset)
            lca = lcaScore(t, a, b)
            d = torch.linalg.norm(ea - eb)
            lcaScores.append(lca)
            dScores.append(d)
        return torch.tensor(lcaScores), torch.stack(dScores).cpu()

    def plot_data (self, win, dataset) : 
        lca, dlca = self.sample(dataset)
        unique = torch.unique(lca)
        cols = []
        maxSize = 0
        for u in unique : 
            cols.append(dlca[lca == u])
            maxSize = max(dlca[lca == u].shape[0], maxSize)
        for i in range(len(cols)) : 
            col = cols[i]
            while col.shape[0] < maxSize: 
                col = torch.cat((col, col), 0)
            col = col[:maxSize]
            cols[i] = col
        data = torch.stack(cols).squeeze().t()
        data = torch.log(data)
        legend = [str(float(u)) for u in unique]
        opts = dict(
            legend=legend, 
            xlabel="LCADistance",
            ylabel="log(Distance)",
            title=win
        )
        self._api.boxplot(data, win=win, opts=opts)

    def batch_end(self, batch, step_data) :
        super(ConfusionDistanceCallback, self).batch_end(batch, step_data)
        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0
        self.plot_data("confusion_distance", self.dataset)
        self._step += 1

    def val_batch_end (self, batch_data, running_val_data) : 
        super(ConfusionDistanceCallback, self).val_batch_end(batch_data, running_val_data)
        self.plot_data("val_confusion_distance", self.val_dataset) 
        self._step += 1

class SchedulerCallback (ttools.callbacks.Callback) : 

    def __init__ (self, sched) :
        super(SchedulerCallback, self).__init__()
        self.sched = sched

    def epoch_end (self) :
        super(SchedulerCallback, self).epoch_end()
        self.sched.step()

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
        self._step = 0
        self.frequency = frequency

    def epoch_start (self, epoch) : 
        super(ConfusionLineCallback, self).epoch_start(epoch)
        self.ratio = None
        self.dratio = None 

    def plot_data (self, ratio, dratio, win) : 
        unique = torch.unique(ratio)
        cols = []
        maxSize = 0
        for u in unique : 
            cols.append(dratio[ratio == u])
            maxSize = max(dratio[ratio == u].shape[0], maxSize)
        for i in range(len(cols)) : 
            col = cols[i]
            while col.shape[0] < maxSize: 
                col = torch.cat((col, col), 0)
            col = col[:maxSize]
            cols[i] = col
        data = torch.stack(cols).squeeze().t()
        data = torch.log(data)
        legend = [str(float(u)) for u in unique]
        opts = dict(
            legend=legend, 
            xlabel="(LCAminus / LCAplus)",
            ylabel="log(dminus / dplus)",
            title=win
        )
        self._api.boxplot(data, win=win, opts=opts)

    def batch_end(self, batch, step_data) :
        super(ConfusionLineCallback, self).batch_end(batch, step_data)
        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0
        refPlus = batch['refPlus'].cpu().detach()
        refMinus = batch['refMinus'].cpu().detach()
        ratio = refMinus / refPlus
        dratio = step_data['dratio'].cpu().detach()
        self.ratio = ratio if self.ratio is None else torch.cat([ratio, self.ratio])
        self.dratio = dratio if self.dratio is None else torch.cat([dratio, self.dratio])
        self.plot_data(self.ratio, self.dratio, "confusion")
        self._step += 1

    def val_batch_end (self, batch_data, running_val_data) : 
        super(ConfusionLineCallback, self).val_batch_end(batch_data, running_val_data)
        ratio = running_val_data['ratio'].cpu().detach()
        dratio = running_val_data['dratio'].cpu().detach()
        self.plot_data(ratio, dratio, "val_confusion") 
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
            n = math.isqrt(N)
            viz =  torch.stack([torch.cat(chunks[i*n: (i+1)*n], 1) for i in range(n)])
            return viz

    def caption(self, batch, step_data, is_val):
        if not is_val: 
            return self.key

class ImageCallback(ttools.callbacks.ImageDisplayCallback):
    def visualized_image(self, batch, step_data, is_val):
        im = batch['im'][0].cpu().unsqueeze(0)
        im = torch.cat((im, im), 2)
        refCrop = batch['refCrop'][0].cpu().unsqueeze(0)
        refWhole = batch['refWhole'][0].cpu().unsqueeze(0)
        ref = torch.cat((refCrop, refWhole), 2)
        plusCrop = batch['plusCrop'][0].cpu().unsqueeze(0)
        plusWhole = batch['plusWhole'][0].cpu().unsqueeze(0)
        plus = torch.cat((plusCrop, plusWhole), 2)
        minusCrop = batch['minusCrop'][0].cpu().unsqueeze(0)
        minusWhole = batch['minusWhole'][0].cpu().unsqueeze(0)
        minus = torch.cat((minusCrop, minusWhole), 2)
        viz = torch.cat([im, ref, plus, minus], 3)
        viz = (viz - viz.min()) / (viz.max() - viz.min())
        return viz
        
    def caption(self, batch, step_data, is_val):
        # write some informative caption into the visdom window
        refMinus = int(batch['refMinus'][0].cpu())
        refPlus = int(batch['refPlus'][0].cpu())
        return f'{refMinus} / {refPlus}'

