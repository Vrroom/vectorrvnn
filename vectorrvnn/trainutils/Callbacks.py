import torch
from torch import nn
from sklearn import metrics
from functools import partial
from vectorrvnn.utils import *
from .torchTools import * 
import numpy as np
from ttools.callbacks import *
import visdom

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

    def node2Image (self, node, mask) : 
        ims = tensorFilter(node, isImage)
        if mask is None : 
            ims = [im[0] for im in ims]
        else : 
            try : 
                mask = mask.view(-1)
                ims = [im[mask][0] for im in ims]
            except Exception : 
                ims = [torch.ones_like(im[0]) for im in ims]
        return torch.stack(ims)

    def visualized_image(self, batch, step_data, is_val):
        mask = step_data['mask']
        ref   = self.node2Image(batch['ref'], mask)
        plus  = self.node2Image(batch['plus'], mask)
        minus = self.node2Image(batch['minus'], mask)
        viz = torch.cat([ref, plus, minus], 3)
        return viz
        
    def caption(self, batch, step_data, is_val):
        return 'triplets'

class FMICallback (Callback) : 
    """ Plot fmi for the validation set after each epoch """
    def __init__(self, model, valData, frequency=100, server=None,
            port=8097, base_url="/", env="main", log=False):
        super(FMICallback, self).__init__()
        if server is None:
            server = "http://localhost"
        self._api = visdom.Visdom(server=server, port=port, env=env,
                                  base_url=base_url)
        if self._api.win_exists("val_FMI"):
            self._api.close("val_FMI")
        self.model = model
        self.valData = valData
        self.frequency = frequency

    def validation_start(self, dataloader) : 
        super(FMICallback, self).validation_start(dataloader)
        data = filter(lambda t: t.nPaths < 40, self.valData)
        data = list(map(forest2tree, data))
        out = list(map(self.model.greedyTree, data))
        # FIXME : Find out how to plot all of them.
        scores = list(map(
            lambda k : avg(map(partial(fmi, level=k), data, out)),
            range(1, 2)
        ))
        t = self.epoch + 1
        for i, s in enumerate(scores) : 
            opts=dict(
                title=f'FMI-{i}', 
                ylabel=f'FMI-{i}',
                xlabel="Epoch"
            )
            self._api.line([s], [t], win="val_FMI", update="append", name="", opts=opts)


