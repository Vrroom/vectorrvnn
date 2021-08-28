import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import random
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from vectorrvnn.network import *
from vectorrvnn.baselines import * 
from functools import partial
import os
import ttools 
import ttools.interfaces
from ttools.callbacks import *
from ttools.modules import networks
from copy import deepcopy
from tqdm import tqdm

LOG = ttools.get_logger(__name__)

class TripletInterface (ttools.ModelInterface) : 

    def __init__(self, opts, model, dataset, val_dataset, 
            cuda=True, max_grad_norm=10):
        super(TripletInterface, self).__init__()
        self.opts = opts
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.epoch = 0
        self.dataset = dataset
        self.val_dataset = val_dataset
        trainedParams = filter(lambda p: p.requires_grad, self.model.parameters())
        self.opt = optim.Adam(trainedParams, lr=opts.lr, weight_decay=opts.wd)
        self.sched = getScheduler(self.opt, opts)
        self.init = deepcopy(self.model.state_dict())

    def _accumulate(self, data, running_data) : 
        oc, nc = running_data['count'], data['count']
        combiners = dict(
            count=lambda x, y : x,
            loss=lambda x, y : (x + oc * y) / nc,
            mask=lambda x, y : torch.cat((x, y), dim=0),
            dplus=lambda x, y : torch.cat((x, y), dim=0),
            dminus=lambda x, y : torch.cat((x, y), dim=0),
            hardpct=lambda x, y: (x + oc * y) / nc
        )
        ret = {}
        for k, fn in combiners.items() : 
            d, rd = data[k], running_data[k]
            if d is None : 
                ret[k] = None
            elif rd is None : 
                ret[k] = d
            else : 
                ret[k] = fn(d, rd)
        return ret

    def _clip_gradients (self) : 
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)

    def _log_wt_stats (self, ret) : 
        with torch.no_grad() : 
            for name, module in self.model.named_children() : 
                wd = avg(map(
                    lambda x : x.pow(2).mean(), 
                    module.parameters())
                )
                ret[f'{name}_wd'] = wd

    def _log_lr (self, ret) : 
        lr = self.opt.state_dict()['param_groups'][0]['lr']
        ret['lr'] = lr

    def training_step(self, batch) :
        self.model.train()
        ret = self.model(**batch)
        loss = ret['loss']
        # optimize
        self.opt.zero_grad()
        loss.backward()
        self._clip_gradients()
        self.opt.step()
        self._log_lr(ret)
        self._log_wt_stats(ret)
        # convert all tensor scalars to scalars
        tensorApply(
            ret, 
            lambda t : t.item(), 
            lambda t : t.nelement() == 1
        )
        return ret

    def init_validation(self):
        return dict(
            loss=0,
            mask=None,
            dplus=None,
            dminus=None,
            hardpct=None,
            count=0
        )

    def validation_step(self, batch, running_data) : 
        self.model.eval()
        with torch.no_grad():
            ret = self.model(**batch)
            ret['count'] = 1 + running_data['count']
            ret = self._accumulate(ret, running_data)
            tensorApply(
                ret, 
                lambda t : t.item(), 
                lambda t : t.nelement() == 1
            )
            return ret


def addCallbacks (trainer, model, data, opts) : 
    modelParams = [n for n, _ in model.named_children()]
    weightDecay = [f'{n}_wd' for n in modelParams]
    keys = ["loss", "hardpct", "lr", *weightDecay]
    _, valData, trainDataLoader, _ = data
    trainer.add_callback(
        SchedulerCallback(trainer.interface.sched)
    )
    checkpointer = ttools.Checkpointer(
        osp.join(opts.checkpoints_dir, opts.name),
        model
    )
    trainer.add_callback(
        CheckpointingCallback(checkpointer)
    )
    trainer.add_callback(
        ProgressBarCallback(
            keys=keys, 
            val_keys=keys[:1]
        )
    )
    trainer.add_callback(
        TripletVisCallback(
            env=opts.name + "_vis", 
            win="samples", 
            frequency=opts.frequency
        )
    )
    trainer.add_callback(
        VisdomLoggingCallback(
            keys=keys, 
            val_keys=keys[:2], 
            env=opts.name + "_training_plots", 
            frequency=opts.frequency
        )
    )
    trainer.add_callback(
        KernelDisplayCallback(
            model,
            win="kernel", 
            env=opts.name + "_kernel", 
            frequency=opts.frequency
        )
    )
    trainer.add_callback(
        TreeScoresCallback(
            model, 
            data,
            frequency=opts.frequency,
            env=opts.name + "_treeScores"
        )
    )
    trainer.add_callback(
        GradientLoggingCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_gradients"
        )
    )
    trainer.add_callback(
        HierarchyVisCallback(
            model,
            data,
            frequency=opts.frequency,
            env=opts.name + "_hierarchy"
        )
    )
    for vis in model.vis : 
        trainer.add_callback(vis)
    trainer.add_callback(
        CheckpointingBestNCallback(checkpointer, key='fmi')
    )

def buildModel (opts) : 
    # Load pretrained path module
    ModelCls = globals()[opts.modelcls]
    model = ModelCls(opts).float()
    if opts.load_ckpt is not None : 
        initPath = osp.join(
            opts.checkpoints_dir, 
            opts.load_ckpt
        )
        state_dict = torch.load(initPath)
        model.load_state_dict(state_dict['model'], strict=False)
    model.to(opts.device)
    if opts.phase == 'train' : 
        model.train()
    else :
        model.eval()
    print(model)
    return model

def buildData (opts) : 
    trainData = TripletDataset(osp.join(opts.dataroot, 'Train'))
    valData = TripletDataset(osp.join(opts.dataroot, 'Val'))
    SamplerCls = globals()[opts.samplercls]
    trainDataLoader = TripletDataLoader(
        opts=opts, 
        sampler=SamplerCls(
            trainData, 
            opts.train_epoch_length,
            transform=getGraphicAugmentation(opts)
        )
    )
    valDataLoader = TripletDataLoader(
        opts=opts, 
        sampler=SamplerCls(
            valData,
            opts.val_epoch_length,
            val=True
        )
    )
    return trainData, valData, trainDataLoader, valDataLoader

def train (opts) : 
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader = data
    model = buildModel(opts) 
    model.train()
    interface = TripletInterface(opts, model, trainData, valData)
    trainer = ttools.Trainer(interface)
    addCallbacks(trainer, model, data, opts)
    # Start training
    trainer.train(
        trainDataLoader, 
        num_epochs=opts.n_epochs, 
        val_dataloader=valDataLoader
    )

def logScores (ts1, ts2, methodName, fd) : 
    scoreFn = lambda t, t_ : cted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    tedscore = avg(map(scoreFn, ts1, ts2))
    fmi1score = avg(map(partial(fmi, level=1), ts1, ts2))
    fmi2score = avg(map(partial(fmi, level=2), ts1, ts2))
    fmi3score = avg(map(partial(fmi, level=3), ts1, ts2))
    fd.write(methodName + '\n')
    fd.write(f'C.T.E.D.  = {tedscore}\n')
    fd.write(f'F.M.I.(1) = {fmi1score}\n')
    fd.write(f'F.M.I.(2) = {fmi2score}\n')
    fd.write(f'F.M.I.(3) = {fmi3score}\n')

def test (opts) : 
    testData = TripletDataset(osp.join(opts.dataroot, 'Test'))
    trainData = TripletDataset(osp.join(opts.dataroot, 'Train'))
    graphics = [repr(t.doc) for t in trainData]
    model = buildModel(opts)
    ts1 = list(map(forest2tree, testData))
    ts1 = [_ for _ in ts1 if repr(_.doc) not in graphics]
    ts2 = list(map(model.greedyTree, ts1))
    ts3 = list(map(model.containmentGuidedTree, ts1))
    ts4 = list(map(autogroup, ts1))
    ts5 = list(map(suggero, ts1))
    exprDir = osp.join(opts.checkpoints_dir, opts.name)
    logFile = osp.join(exprDir, f'{opts.name}.log')
    with open(logFile, 'w+') as fd : 
        logScores(ts1, ts2, "Ours (greedyTree)", fd)
        logScores(ts1, ts3, "Ours (containment guided)", fd)
        logScores(ts1, ts4, "Autogroup", fd)
        logScores(ts1, ts5, "Suggero", fd)

if __name__ == "__main__" : 
    opts = Options().parse()
    if opts.phase == 'train' : 
        train(opts)
    elif opts.phase == 'test' : 
        test(opts)

