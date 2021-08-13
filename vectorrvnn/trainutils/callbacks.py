import torch
from torch import nn
from torch.optim.swa_utils import *
from datetime import datetime
from sklearn import metrics
from functools import partial
from itertools import starmap
from more_itertools import unzip
from vectorrvnn.utils import *
from .torchTools import * 
import numpy as np
from ttools.callbacks import *
from ttools.utils import *
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
    def __init__ (self, model, data, frequency=100, env="main") : 
        super(HierarchyVisCallback, self).__init__() 
        self._api = visdom.Visdom(env=env)
        self.nWindows = 10
        for i in range(self.nWindows) : 
            closeWindow(self._api, f'hierarchies-{i}')
        self.model = model
        trainData, valData, _, _  = data
        trainGraphics = [t.svgFile for t in trainData]
        self.valData = [v for v in valData if v.svgFile not in trainGraphics]
        self.frequency = frequency
        self.rng = random.Random(0)

    def validation_start (self, dataloader) : 
        super(HierarchyVisCallback, self).validation_start(dataloader)
        data = self.rng.sample(self.valData, k=self.nWindows) 
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

    def _bbox_df (self, batch, mask, node) : 
        if mask is None : 
            bbox = batch[node]['bbox'][0]
        else : 
            mask = mask.view(-1)
            bbox = batch[node]['bbox'][mask][0]
        bbox = bbox.view(-1).detach().cpu().numpy()
        x, y, w, h = bbox
        y = 1 - y 
        df = pd.DataFrame(data=dict(
            x=[x, x, x + w, x + w, x],
            y=[y, y - h, y - h, y, y],
            nodeType=[node] * 5,
        ))
        return df

    def _plot_bbox (self, batch, mask, win) : 
        refBox   = self._bbox_df(batch, mask, 'ref')
        plusBox  = self._bbox_df(batch, mask, 'plus')
        minusBox = self._bbox_df(batch, mask, 'minus')
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
        mask = step_data['mask']
        self._tstep = 1
        self._plot_bbox(batch, mask, 'bbox-train')

    def val_batch_end (self, batch, running_data)  :
        super(BBoxVisCallback, self).val_batch_end(batch, running_data)
        if self._vstep % self.frequency != 0:
            self._vstep += 1
            return
        self._vstep = 1
        self._plot_bbox(batch, 'bbox-val')

class TreeScoresCallback (Callback) : 
    """ Plot fmi and cted for the validation set after each epoch """
    def __init__(self, model, data, frequency=100, env="main"):
        super(TreeScoresCallback, self).__init__()
        self._api = visdom.Visdom(env=env)
        closeWindow(self._api, 'val_TreeScores')
        self.model = model
        trainData, valData, _, _  = data
        trainGraphics = [t.svgFile for t in trainData]
        self.valData = [v for v in valData if v.svgFile not in trainGraphics]
        self.frequency = frequency

    def validation_end(self, val_data) : 
        super(TreeScoresCallback, self).validation_end(val_data)
        data = list(filter(lambda t: t.nPaths < 40, self.valData))
        out = list(map(self.model.greedyTree, data))
        scores = list(map(
            lambda k : avg(map(partial(fmi, level=k), data, out)),
            range(1, 4)
        ))
        normedCted = lambda x, y : cted(x, y) / (x.number_of_nodes() + y.number_of_nodes())
        ctedScore = avg(map(normedCted, data, out))
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
            [ctedScore],
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

class GradientLoggingCallback (Callback) : 
    """ Plot gradients of all parameters """

    def __init__ (self, model, frequency=100, server="http://localhost", 
            port=8097, env="main", base_url="/", smoothing=0.9) : 
        super(GradientLoggingCallback, self).__init__()
        self.model = model 
        self._api = visdom.Visdom(server=server, port=port, 
            env=env, base_url=base_url)

        self.win = "gradients"
        closeWindow(self._api, self.win)

        self.keys = list(unzip(model.named_modules())[0])
        legend = self.keys
        self._opts = {
            "legend": legend,
            "title": self.win,
            "xlabel": "epoch",
        }
        self._step = 0
        self.frequency = frequency
        self.ema = ExponentialMovingAverage(self.keys, alpha=smoothing)

    def batch_end(self, batch, train_step_data):
        super(GradientLoggingCallback, self).batch_end(batch, train_step_data)

        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0

        t = self.batch / max(self.datasize, 1) + self.epoch

        modules = list(unzip(self.model.named_modules())[1])
        grads = list(map(moduleGradNorm, modules))
        for k, g in zip(self.keys, grads) : 
            self.ema.update(k, g)
        data = np.array([self.ema[k] for k in self.keys])
        data = np.expand_dims(data, 1)
        self._api.line(data, [t], update="append", win=self.win, opts=self._opts)

        self._step += 1
