import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from vectorrvnn.network import *
from vectorrvnn.baselines import * 
import os
import ttools 
import ttools.interfaces
from ttools.callbacks import *
from ttools.modules import networks
from subprocess import call
from copy import deepcopy
from functools import partial

def firstDimCat(x, y, r) :
    return torch.cat((x, y), dim=0)

def movingAvg (x, y, r) : 
    nc = r['count']
    oc = nc - 1
    return (x + oc * y) / nc

def noneSkipper (fn, a, b, r) :
    if a is None:
        return None
    elif b is None :
        return a
    else :
        return fn(a, b, r)

def dictAccumulate(combiner, b, a) : 
    c = {}
    c['count'] = b['count'] + 1
    for k, fn in combiner.items() : 
        d, rd = a[k], b[k]
        c[k] = noneSkipper(fn, d, rd, c)
    return c

class Interface (ttools.ModelInterface) : 

    def __init__(self, opts, model, dataset, 
            val_dataset, cuda=True, max_grad_norm=10):
        super(Interface, self).__init__()
        self.opts = opts
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.epoch = 0
        self.dataset = dataset
        self.val_dataset = val_dataset
        trainedParams = filter(lambda p: p.requires_grad, self.model.parameters())
        self.opt = optim.Adam(
            trainedParams, 
            lr=opts.lr, 
            weight_decay=opts.wd
        )
        self.init = deepcopy(self.model.state_dict())
        self.combiners = dict(
            loss=movingAvg,
            mask=firstDimCat,
            dplus=firstDimCat,
            dminus=firstDimCat,
            hardpct=movingAvg
        )


    def training_step(self, batch) :
        self.model.train()
        self.opt.zero_grad()
        ret = self.model(**batch)
        ret['loss'].backward()
        clipGradients(self.model, self.max_grad_norm)
        self.opt.step()
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
            ret = dictAccumulate(self.combiners, running_data, ret)
            tensorApply(
                ret, 
                lambda t : t.item(), 
                lambda t : t.nelement() == 1
            )
            return ret

def buildModel (opts) : 
    # Load pretrained path module
    ModelCls = opts.modelcls
    model = ModelCls(opts).float()
    if opts.load_ckpt is not None : 
        initPath = osp.join(
            opts.checkpoints_dir, 
            opts.load_ckpt
        )
        state_dict = torch.load(initPath, map_location=opts.device)
        model.load_state_dict(state_dict['model'])
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
        if osp.exists(d) : 
            call(['rm', '-r', d])
    for d in dirs: 
        mkdir(d)
    total = len(allData)
    tPt, vPt = int(0.6 * total), int(0.2 * total)
    datasets = [allData[:tPt], allData[tPt:tPt + vPt], allData[tPt + vPt:]]
    for dir, dataset in zip(dirs, datasets) : 
        [call(['cp', '-r', f, dir]) for f in dataset]
    return dirs

def buildDataLoader (trainData, valData, opts) : 
    SamplerCls = globals()[opts.samplercls]
    DataloaderCls = globals()[opts.dataloadercls]
    trainDataLoader = DataloaderCls(
        opts=opts, 
        sampler=SamplerCls(
            trainData, 
            opts.train_epoch_length,
            opts,
            transform=getGraphicAugmentation(opts)
        )
    )
    valDataLoader = DataloaderCls(
        opts=opts, 
        sampler=SamplerCls(
            valData,
            opts.val_epoch_length,
            opts,
            transform=getGraphicAugmentation(opts),
            val=True
        )
    )
    return trainDataLoader, valDataLoader

def buildData (opts) : 
    dataDirs = getDataSplits(opts)
    trainData = SVGDataset(dataDirs[0])
    valData   = SVGDataset(dataDirs[1], trainData.ids)
    testData  = SVGDataset(dataDirs[2], trainData.ids)
    trainDataLoader, valDataLoader = buildDataLoader(trainData, valData, opts)
    return trainData, valData, trainDataLoader, valDataLoader, testData

def setSeed (opts) : 
    rng.seed(opts.seed)
    print("Set rng seed to -", opts.seed)
    print("First random int -", rng.randint(0, 1000))

def addGenericCallbacks(trainer, model, data, opts) : 
    keys = ["loss", "hardpct"]
    _, valData, trainDataLoader, _, _ = data
    checkpointer = ttools.Checkpointer(
        osp.join(opts.checkpoints_dir, opts.name),
        model
    )
    trainer.add_callback(CheckpointingCallback(checkpointer))
    trainer.add_callback(
        ProgressBarCallback(keys=keys, val_keys=keys[:1])
    )
    trainer.add_callback(
        LRCallBack(
            trainer.interface.opt,
            env=opts.name + "_lr"
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
    # trainer.add_callback(
    #     HierarchyVisCallback(
    #         model,
    #         valData,
    #         opts,
    #         env=opts.name + "_hierarchy"
    #     )
    # )
    trainer.add_callback(
        TreeScoresCallback(
            model, 
            valData,
            opts,
            env=opts.name + "_tree_scores"
        )
    )
    # trainer.add_callback(
    #     SiblingEmbeddingsCallback(
    #         model,
    #         valData, 
    #         opts,
    #         env=opts.name + "_sib"
    #     )
    # )
    # trainer.add_callback(
    #     VisHardestCallback(
    #         model,
    #         valData, 
    #         opts,
    #         env=opts.name + "_hardest"
    #     )
    # )
    trainer.add_callback(
        DistanceHistogramCallback(
            frequency=opts.frequency,
            env=opts.name + "_distance"
        )
    )
    trainer.add_callback(
        CheckpointingBestNCallback(checkpointer, key='fmi')
    )
    trainer.add_callback(
        GradCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_gradients"
        )
    )
    # trainer.add_callback(
    #     InitDistanceCallback(
    #         model,
    #         frequency=opts.frequency,
    #         env=opts.name + "_init_distance"
    #     )
    # )
    trainer.add_callback(
        NormCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_norms"
        )
    )

def train (opts, callbackFn) : 
    data = buildData(opts)
    trainData, valData, trainDataLoader, valDataLoader, _ = data
    model = buildModel(opts) 
    model.train()
    interface = Interface(opts, model, trainData, valData)
    trainer = ttools.Trainer(interface)
    callbackFn(trainer, model, data, opts)
    # Start training
    trainer.train(
        trainDataLoader, 
        num_epochs=opts.n_epochs, 
        val_dataloader=valDataLoader
    )

