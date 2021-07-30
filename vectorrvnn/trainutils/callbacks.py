import torch
from torch import nn
from torch.optim.swa_utils import *
from datetime import datetime
from sklearn import metrics
from functools import partial
from itertools import starmap
from vectorrvnn.utils import *
from .torchTools import * 
import numpy as np
from ttools.callbacks import *
import ttools
import visdom
import random
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

def closeWindow (api, name) : 
    if api.win_exists(name) : 
        api.close(name)

class SchedulerCallback (Callback) : 
    """ Take a step of the lr schedular after each epoch end """
    def __init__ (self, sched) :
        super(SchedulerCallback, self).__init__()
        self.sched = sched

    def epoch_end (self) :
        super(SchedulerCallback, self).epoch_end()
        self.sched.step()

class SWACallback (SchedulerCallback) : 
    """ Manage stochastic weight averaging """
    def __init__ (self, sched, model, swaModel, dataloader, opts) : 
        super(SWACallback, self).__init__(sched)
        self.model = model
        self.swaModel = swaModel
        self.dataloader = dataloader
        self.opts = opts

    def epoch_end (self) : 
        self.swaModel.update_parameters(self.model)
        super(SWACallback, self).epoch_end()

    def training_end (self) : 
        """ Set BatchNorm Stats """ 
        super(SWACallback, self).training_end()
        for batch in self.dataloader:  
            tensorApply(
                batch,
                lambda t : t.to(self.opts.device)
            )
            self.swaModel(**batch)

class KernelDisplayCallback (ImageDisplayCallback) : 
    """ display the kernels of the first convolutional layer """
    def __init__ (self, model, env, win, frequency=100) : 
        super(KernelDisplayCallback, self).__init__(
            env=env, 
            win=win, 
            frequency=frequency
        )
        self.model = model

    def visualized_image(self, batch, step_data, is_val):
        if not is_val : 
            first_module = next(filter(
                lambda x: isinstance(x, nn.Conv2d), 
                self.model.modules()
            ))
            kernels = first_module.weight.cpu()
            N, *_ = kernels.shape
            chunks = torch.chunk(kernels, N)
            chunks = [c.squeeze() for c in chunks]
            n = int(np.floor(np.sqrt(N)))
            viz =  torch.stack([torch.cat(chunks[i*n: (i+1)*n], 1) for i in range(n)])
            viz = viz[:, :3, :, :]
            viz = (viz - viz.min()) / (viz.max() - viz.min())
            return viz

    def caption(self, batch, step_data, is_val):
        if not is_val: 
            return 'kernel'

class TripletVisCallback(ImageDisplayCallback): 
    """ visualize the images for each node in triplet """
    def node2Image (self, node, mask, i=0) : 
        ims = tensorFilter(node, isImage)
        tensorApply(ims, toGreyScale, isGreyScale)
        if mask is None : 
            ims = [im[i] for im in ims]
        else : 
            try : 
                mask = mask.view(-1)
                ims = [im[mask][i] for im in ims]
            except Exception : 
                ims = [torch.ones_like(im[i]) for im in ims]
        tensorApply(ims, normalize2UnitRange)
        return torch.cat(ims, 2)

    def visualized_image(self, batch, step_data, is_val, i=0):
        mask = step_data['mask']
        ref   = self.node2Image(batch['ref'], mask, i)
        plus  = self.node2Image(batch['plus'], mask, i)
        minus = self.node2Image(batch['minus'], mask, i)
        viz = torch.stack([ref, plus, minus])
        return viz
        
    def caption(self, batch, step_data, is_val):
        return 'ref, plus, minus'

class DistanceHistogramCallback (Callback) : 
    """ show the distribution of dplus and dminus """
    def __init__ (self, frequency=100, env="main") : 
        super(DistanceHistogramCallback, self).__init__() 
        self._api = visdom.Visdom(env=env)
        closeWindow(self._api, 'distance-train')
        closeWindow(self._api, 'distance-val')
        self.frequency = frequency

    def _plot_distribution (self, dplus, dminus, win) : 
        labels = (['plus'] * len(dplus) + ['minus'] * len(dminus))
        df = pd.DataFrame(data=dict(
            distance=np.concatenate((dplus, dminus)),
            labels=labels
        ))
        fig = px.histogram(
            df, 
            x="distance", 
            color="labels",
            opacity=0.5,
            nbins=100
        )
        self._api.plotlyplot(fig, win=win)

    def batch_end (self, batch, step_data) : 
        super(DistanceHistogramCallback, self).batch_end(batch, step_data)
        dplus = step_data['dplus'].view(-1).detach().cpu().numpy()
        dminus = step_data['dminus'].view(-1).detach().cpu().numpy()
        self._plot_distribution(dplus, dminus, 'distance-train')

    def val_batch_end (self, batch, running_data)  :
        super(DistanceHistogramCallback, self).val_batch_end(batch, running_data)
        dplus = running_data['dplus'].view(-1).detach().cpu().numpy()
        dminus = running_data['dminus'].view(-1).detach().cpu().numpy()
        self._plot_distribution(dplus, dminus, 'distance-val')

class HierarchyVisCallback (Callback) : 
    """ show inferred hierarchies """
    def __init__ (self, model, valData, frequency=100, env="main") : 
        super(HierarchyVisCallback, self).__init__() 
        self._api = visdom.Visdom(env=env)
        self.nWindows = 10
        for i in range(self.nWindows) : 
            closeWindow(self._api, f'hierarchies-{i}')
        self.model = model
        self.valData = valData
        self.frequency = frequency
        self.rng = random.Random(0)

    def validation_start (self, dataloader) : 
        super(HierarchyVisCallback, self).validation_start(dataloader)
        data = self.rng.sample(list(self.valData), k=self.nWindows) 
        data = list(map(forest2tree, data))
        out = list(map(self.model.greedyTree, data))
        for i in range(self.nWindows) : 
            out[i].doc = data[i].doc
            fig, ax = plt.subplots(1, 1, dpi=100) 
            treeAxisFromGraph(out[i], fig, ax)
            self._api.matplot(plt, win=f'hierarchies-{i}')
            plt.close()

class BBoxVisCallback (Callback) : 

    def __init__ (self, frequency=100, env="main") :
        super(BBoxVisCallback, self).__init__()
        self._api = visdom.Visdom(env=env)
        closeWindow(self._api, 'bboxes')
        self._tstep = 0
        self._vstep = 0
        self.frequency = frequency

    def _bbox_df (self, batch, node) : 
        bbox = batch[node]['bbox'][0].view(-1).detach().cpu().numpy() 
        x, y, w, h = bbox
        y = 1 - y 
        df = pd.DataFrame(data=dict(
            x=[x, x, x + w, x + w, x],
            y=[y, y - h, y - h, y, y],
            nodeType=[node] * 5,
        ))
        return df

    def _plot_bbox (self, batch, win) : 
        refBox   = self._bbox_df(batch, 'ref')
        plusBox  = self._bbox_df(batch, 'plus')
        minusBox = self._bbox_df(batch, 'minus')
        df = pd.concat((refBox, plusBox, minusBox))
        fig = px.line(
            df, 
            x="x", 
            y="y",
            color="nodeType",
            title=win
        )
        fig.update_xaxes(range=[0, 1])
        fig.update_yaxes(range=[0, 1])
        self._api.plotlyplot(fig, win=win)

    def batch_end (self, batch, step_data) : 
        super(BBoxVisCallback, self).batch_end(batch, step_data)
        if self._tstep % self.frequency != 0:
            self._tstep += 1
            return
        self._tstep = 1
        self._plot_bbox(batch, 'bbox-train')

    def val_batch_end (self, batch, running_data)  :
        super(BBoxVisCallback, self).val_batch_end(batch, running_data)
        if self._vstep % self.frequency != 0:
            self._vstep += 1
            return
        self._vstep = 1
        self._plot_bbox(batch, 'bbox-val')

class TreeScoresCallback (Callback) : 
    """ Plot fmi and cted for the validation set after each epoch """
    def __init__(self, model, valData, frequency=100, env="main"):
        super(TreeScoresCallback, self).__init__()
        self._api = visdom.Visdom(env=env)
        closeWindow(self._api, 'val_TreeScores')
        self.model = model
        self.valData = valData
        self.frequency = frequency

    def validation_end(self, val_data) : 
        super(TreeScoresCallback, self).validation_end(val_data)
        data = filter(lambda t: t.nPaths < 40, self.valData)
        data = list(map(forest2tree, data))
        out = list(map(self.model.greedyTree, data))
        scores = list(map(
            lambda k : avg(map(partial(fmi, level=k), data, out)),
            range(1, 4)
        ))
        ctedScore = avg(map(cted, data, out))
        t = self.epoch + 1
        opts=dict(
            title='TreeScores', 
            ylabel='TreeScores',
            xlabel="Epoch"
        )
        for i, s in enumerate(scores) : 
            self._api.line(
                [s], 
                [t], 
                win="val_TreeScores", 
                update="append", 
                name=f'FMI-{i}', 
                opts=opts
            )
        self._api.line(
            [s],
            [t],
            win="val_TreeScores", 
            update="append",
            name="cted",
            opts=opts
        )
        val_data['fmi'] = scores[0]

class CheckpointingBestNCallback (Callback) : 
    """ A callback which saves the best N models.

    Args:
        checkpointer (Checkpointer): actual checkpointer responsible for the I/O
        key: key into accumulated validation data to define metric.
        N (int, optional): number of models to save
        sense (string, optional): one of "maximize"/"minimize". 
            Denoting whether we want to maximize/minimize val metric.
    """
    
    BEST_PREFIX = "best_"

    def __init__ (self, checkpointer, key, N=3, sense="maximize") : 
        super(CheckpointingBestNCallback, self).__init__()
        self.checkpointer = checkpointer
        self.key = key
        self.N = N
        self.sense = sense
        self.default = sense == "maximize"
        self.cmp = lambda x, y : x > y if self.default else y > x
        self.ckptDict = dict()

    def validation_end(self, val_data): 
        super(CheckpointingBestNCallback, self).validation_end(val_data)
        score = val_data[self.key] 
        isBetter = any([self.cmp(score, y) for y in self.ckptDict.keys()])
        if len(self.ckptDict) < self.N or isBetter : 
            path = "{}{:.3f}".format(CheckpointingBestNCallback.BEST_PREFIX, score)
            path = path.replace('.', '-')
            path = path + '-' + datetime.now().strftime('%m-%d-%Y-%H-%M-%S')
            self.checkpointer.save(path, extras=dict(score=score))
            self.ckptDict[score] = path
            self.__purge_old_files()

    def __purge_old_files(self) : 
        """Delete checkpoints that are beyond the max to keep."""
        chkpts = os.listdir(self.checkpointer.root)
        toBeRemoved = sorted(self.ckptDict.keys(), reverse=self.default)[self.N:]
        for s in toBeRemoved : 
            cpref = self.ckptDict[s]
            cname = [fname for fname in chkpts if cpref in fname].pop()
            self.checkpointer.delete(cname)
            self.ckptDict.pop(s)
