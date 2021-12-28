import torch
import traceback
from torch import nn
from torch.optim.swa_utils import *
from datetime import datetime
from sklearn import metrics
from functools import partial
from itertools import starmap
from more_itertools import unzip, flatten
from vectorrvnn.utils import *
from .torchTools import * 
import numpy as np
from ttools.callbacks import *
from ttools.utils import *
import ttools
import visdom
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from copy import deepcopy

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
            try : 
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
            except Exception :
                pass

class HardTripletCallback (ImageDisplayCallback) : 
    """ Visualize hard triplets 
    
    Here we want to check what are the triplets 
    that the network is unable to disambiguate so
    that we can work towards solving them.
    """

    def node2Image (self, batch, node, mask) :
        nodes = list(getAll(batch, node))
        trees = list(flatten(getAll(nodes, 'tree')))
        pss   = list(flatten(getAll(nodes, 'pathSet')))
        ims = []
        for t, ps, hard in zip(trees, pss, mask) : 
            if hard : 
                ims.append(rasterize(
                    subsetSvg(t.doc, ps),
                    100,
                    100
                ))
        im = alphaComposite(np.vstack(ims))
        im = torch.from_numpy(im)
        im = torch.permute(im, (2, 0, 1))
        return im

    def visualized_image (self, batch, step_data, is_val) : 
        if is_val : 
            dplus  = step_data['dplus']
            dminus = step_data['dminus']
            mask = (dplus > dminus).squeeze()
            ref   = self.node2Image(batch, 'ref', mask)
            plus  = self.node2Image(batch, 'plus', mask)
            minus = self.node2Image(batch, 'minus', mask)
            viz = torch.stack([ref, plus, minus])
            return viz

class TripletVisCallback(ImageDisplayCallback): 
    """ Visualize the images for each node in triplet

    Purpose of this class is to ensure that the inputs
    to the networks are correct. 
    """
    def node2Image (self, batch, node) : 
        nodes = list(getAll(batch, node))
        ims = tensorFilter(nodes, isImage)
        tensorApply(ims, toGreyScale, isGreyScale)
        ims = [im[0] for im in ims]
        tensorApply(ims, normalize2UnitRange)
        return torch.cat(ims, 2)

    def visualized_image(self, batch, step_data, is_val):
        try : 
            ref   = self.node2Image(batch, 'ref', mask)
            plus  = self.node2Image(batch, 'plus', mask)
            minus = self.node2Image(batch, 'minus', mask)
            viz = torch.stack([ref, plus, minus])
            return viz
        except Exception :
            pass
        
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

class BoundingBoxCallback (Callback) :
    def __init__ (self, frequency=100, env="main", win='bboxes') :
        super(BoundingBoxCallback, self).__init__()
        self._api = visdom.Visdom(env=env)
        self.win = win
        closeWindow(self._api, win)
        self._tstep = 0
        self._vstep = 0
        self.frequency = frequency
        self.errorShow = False

    def bbox_df(self, batch, mask, node) :
        raise NotImplementedError

    def plot_bbox (self, batch, mask) : 
        refBox   = self.bbox_df(batch[0], mask, 'ref')
        plusBox  = self.bbox_df(batch[0], mask, 'plus')
        minusBox = self.bbox_df(batch[0], mask, 'minus')
        df = pd.concat((refBox, plusBox, minusBox))
        fig = px.line(
            df, 
            x="x", 
            y="y",
            color="nodeType",
            title=self.win
        )
        fig.update_xaxes(range=[-1, 1])
        fig.update_yaxes(range=[-1, 1])
        self._api.plotlyplot(fig, win=self.win)

    def batch_end (self, batch, step_data) : 
        super(BoundingBoxCallback, self).batch_end(batch, step_data)
        if self._tstep % self.frequency != 0:
            self._tstep += 1
            return
        try : 
            mask = step_data['mask']
            self._tstep = 1
            self.plot_bbox(batch, mask)
        except Exception as e :
            if not self.errorShow : 
                print('Couldn\'t plot')
                print(traceback.format_exc())
                self.errorShow = True

class OBBVis (BoundingBoxCallback) : 

    def bbox_df (self, batch, mask, node) : 
        if mask is None : 
            obbs = batch[node]['obb'][0]
        else : 
            mask = mask.view(-1)
            obbs = batch[node]['obb'][mask][0]
        obbs = obbs.detach().cpu().numpy()
        xs, ys = [], []
        for obb in obbs : 
            x, y, w, h = obb[:4]
            box = BBox(x, y, x + w, y + h, w, h)
            rot = obb[4:8].reshape((2, 2))
            center = obb[8:]
            can = OBB(box, center, rot)
            corners = canonical2corners(can)
            xs.append(corners[:, 0].tolist())
            ys.append(corners[:, 1].tolist())
            xs.append([None])
            ys.append([None])
        xs = reduce(lambda a, b : a + b, xs, [])
        ys = reduce(lambda a, b : a + b, ys, [])
        ns = [node] * len(xs)
        df = pd.DataFrame(data=dict(
            x=xs,
            y=ys,
            nodeType=ns
        ))
        return df

class AABBVis (BoundingBoxCallback) : 

    def bbox_df (self, batch, mask, node) : 
        if mask is None : 
            bbox = batch[node]['bbox'][0]
        else : 
            mask = mask.view(-1)
            bbox = batch[node]['bbox'][mask][0]
        bbox = bbox.view(-1).detach().cpu().numpy()
        x, y, w, h = bbox
        df = pd.DataFrame(data=dict(
            x=[x, x, x + w, x + w, x],
            y=[y, y + h, y + h, y, y],
            nodeType=[node] * 5,
        ))
        return df

class TreeEvalCallback (Callback) : 
    """ Super class for Tree Evaluation Metrics"""
    def __init__ (self, model, data, opts, env="main", wins=[]) :
        super(TreeEvalCallback, self).__init__()
        self._api = visdom.Visdom(env=env)
        for win in wins : 
            closeWindow(self._api, win)
        self.wins = wins
        self.model = model
        self.data = [d for d in data if d.nPaths < opts.max_len]

    def eval_trees (self, trees) : 
        raise NotImplementedError

    def validation_start(self, dataloader) :
        super(TreeEvalCallback, self).validation_start(dataloader)
        self.trees = list(map(self.model.greedyTree, self.data))
        self.eval_trees(self.trees) 

class NodeOverlapCallback (TreeEvalCallback) :
    """ Measure node overlap on PublicDomainVectors """ 
    def __init__ (self, model, data, opts, env="main") :
        super(NodeOverlapCallback, self).__init__(
            model, data, opts, env=env, wins=["node_overlap"])
        self.opts = dict(
            title='Node Overlap', 
            ylabel='NO',
            xlabel="Epoch"
        )

    def eval_trees (self, trees) :
        sc = avg(map(node_overlap, self.data, trees))
        t = self.epoch + 1
        self._api.line(
            [sc], 
            [t], 
            win="node_overlap", 
            update="append", 
            name="no",
            opts=self.opts
        )

class HierarchyVisCallback (TreeEvalCallback) : 
    """ show inferred hierarchies """
    def __init__ (self, model, data, opts, env="main") : 
        wins = [f'hierarchy_{i}' for i in range(10)]
        super(HierarchyVisCallback, self).__init__(
            model, data, opts, env=env, wins=wins) 

    def eval_trees (self, trees) : 
        k = min(len(self.data), len(self.wins))
        data = rng.sample(trees, k=k) 
        for win, pt in zip(self.wins, data) : 
            fig, ax = plt.subplots(1, 1, dpi=100) 
            treeAxisFromGraph(pt, fig, ax)
            self._api.matplot(plt, win=win)
            plt.close()

class TreeScoresCallback (TreeEvalCallback) : 
    """ plot fmi and cted for the validation set after each epoch """
    def __init__(self, model, data, opts, env="main"):
        super(TreeScoresCallback, self).__init__(
            model, data, opts, env=env, wins=["tree_scores"])
        self.opts = dict(
            title='TreeScores', 
            ylabel='TreeScores',
            xlabel="Epoch"
        )
        self.names = ['fmi_1', 'fmi_2', 'fmi_3', 'cted']

    def eval_trees(self, trees) : 
        scores = []
        for k in range(1, 4) : 
            fmik = partial(fmi, level=k)
            sk = avg(map(fmik, self.data, trees))
            scores.append(sk)
        sc = avg(map(norm_cted, self.data, trees))
        scores.append(sc)
        t = self.epoch + 1
        for name, s in zip(self.names, scores) : 
            self._api.line(
                [s], 
                [t], 
                win="tree_scores", 
                update="append", 
                name=name,
                opts=self.opts
            )
        self.scores = scores

    def validation_end (self, val_data) : 
        super(TreeScoresCallback, self).validation_end(val_data)
        val_data['fmi'] = self.scores[0]

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

class ModelStatsCallback (Callback) :
    """ 
    Super class for callbacks that need to plot model statistics.
    """
    def __init__ (self, model, frequency=100, server="http://localhost", 
            port=8097, env="main", win="win", base_url="/", smoothing=0.9) : 
        super(ModelStatsCallback, self).__init__()
        self.model = model 
        self._api = visdom.Visdom(server=server, port=port, 
            env=env, base_url=base_url)

        self.win = win
        closeWindow(self._api, self.win)

        self.trackedParameterDict = dict()
        seen = set()
        for name, module in model.named_children() : 
            for pname, param in module.named_parameters() : 
                if param.requires_grad and not param in seen: 
                    self.trackedParameterDict[str((name, pname))] = param
                    seen.add(param)

        self.keys = list(self.trackedParameterDict.keys())
        legend = self.keys
        self._opts = {
            "legend": legend,
            "title" : self.win,
            "xlabel": "epoch",
        }
        self._step = 0
        self.frequency = frequency
        self.ema = ExponentialMovingAverage(self.keys, alpha=smoothing)

    def model_fn (self, key, param) :
        raise NotImplementedError

    def batch_end(self, batch, train_step_data):
        super(ModelStatsCallback, self).batch_end(batch, train_step_data)

        if self._step % self.frequency != 0:
            self._step += 1
            return
        self._step = 0

        t = self.batch / max(self.datasize, 1) + self.epoch

        for k, p in self.trackedParameterDict.items(): 
            self.ema.update(k, self.model_fn(k, p))

        data = np.array([self.ema[k] for k in self.keys])
        data = np.expand_dims(data, 1)
        self._api.line(
            data, 
            [t], 
            update="append", 
            win=self.win, 
            opts=self._opts
        )

        self._step += 1

class GradCallback (ModelStatsCallback) : 
    """ Plot gradients of all parameters """

    def __init__ (self, model, **kwargs) : 
        super(GradCallback, self).__init__(
            model=model, 
            win="gradients",
            **kwargs
        )

    def model_fn(self, key, p) : 
        return (p.grad.norm() / p.nelement()).item()

class NormCallback (ModelStatsCallback) : 
    """ Plot norm of all parameters """

    def __init__ (self, model, **kwargs) : 
        super(NormCallback, self).__init__(
            model=model, 
            win="norms",
            **kwargs
        )

    def model_fn (self, key, p) : 
        return (p.norm() / p.nelement()).item()

class InitDistanceCallback (ModelStatsCallback) :
    """ Plot distance of parameters from init """

    def __init__ (self, model, **kwargs) :
        super(InitDistanceCallback, self).__init__(
            model,
            win="init_distance",
            **kwargs
        )
        self.params = dict()
        seen = set()
        for name, module in model.named_children() : 
            for pname, param in module.named_parameters() : 
                if param.requires_grad and not param in seen: 
                    self.params[str((name, pname))] = deepcopy(param)
                    seen.add(param)

    def model_fn (self, key, p) :
        norm = (p - self.params[key]).norm()
        return (norm / p.nelement()).item()
