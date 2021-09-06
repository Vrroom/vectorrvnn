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
from subprocess import call

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
    _, valData, trainDataLoader, _, _ = data
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

def getDataSplits (opts) : 
    allData = listdir(opts.dataroot)
    rng.shuffle(allData)
    dirs = ['Train', 'Val', 'Test']
    dirs = [osp.join(f'/tmp/{opts.name}/', _) for _ in dirs]
    for d in dirs: 
        mkdir(d)
    total = len(allData)
    tPt, vPt = int(0.6 * total), int(0.2 * total)
    datasets = [allData[:tPt], allData[tPt:tPt + vPt], allData[tPt + vPt:]]
    for dir, dataset in zip(dirs, datasets) : 
        [call(['cp', '-r', f, dir]) for f in dataset]
    return dirs

def buildData (opts) : 
    dataDirs = getDataSplits(opts)
    trainData, valData, testData = list(map(TripletDataset, dataDirs))
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
    return trainData, valData, trainDataLoader, valDataLoader, testData

def train (opts) : 
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader, _ = data
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

def scores2df (ts1, ts2, methodName) : 
    ctedFn = lambda t, t_ : cted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    ctedscore = avg(map(ctedFn, ts1, ts2))
    fmi1score = avg(map(partial(fmi, level=1), ts1, ts2))
    fmi2score = avg(map(partial(fmi, level=2), ts1, ts2))
    fmi3score = avg(map(partial(fmi, level=3), ts1, ts2))
    return pd.DataFrame(
        data=[[ctedscore, fmi1score, fmi2score, fmi3score]],
        index=[methodName],
        columns=['cted', 'fmi1', 'fmi2', 'fmi3']
    )

def test (opts) : 
    trainData, _, _, _, testData = buildData(opts)
    graphics = [repr(t.doc) for t in trainData]
    model = buildModel(opts)
    ts1 = [_ for _ in testData if repr(_.doc) not in graphics]
    ts2 = list(map(model.greedyTree, ts1))
    ts3 = list(map(model.containmentGuidedTree, ts1))
    ts4 = list(map(autogroup, ts1))
    ts5 = list(map(suggero, ts1))
    exprDir = osp.join(opts.checkpoints_dir, opts.name)
    logFile = osp.join(exprDir, f'{opts.name}.csv')
    dd = scores2df(ts1, ts2, "Ours-DD") 
    cg = scores2df(ts1, ts3, "Ours-CG")
    fi = scores2df(ts1, ts4, "Fisher")
    su = scores2df(ts1, ts5, "Suggero")
    combined = pd.concat([dd, cg, fi, su])
    combined.to_csv(logFile)

def setSeed (opts) : 
    rng.seed(opts.seed)
    print("Set rng seed to -", opts.seed)
    print("First random int -", rng.randint(0, 1000))

if __name__ == "__main__" : 
    opts = Options().parse()
    setSeed(opts)
    if opts.phase == 'train' : 
        train(opts)
    elif opts.phase == 'test' : 
        test(opts)

