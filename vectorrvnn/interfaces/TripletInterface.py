import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
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
        optimcls = getattr(optim, opts.optimcls)
        self.opt = optimcls(
            trainedParams, 
            lr=opts.lr, 
            weight_decay=opts.wd
        )
        self.sched = getScheduler(self.opt, opts)
        self.init = deepcopy(self.model.state_dict())
        self.combiners = dict(
            loss=movingAvg,
            mask=firstDimCat,
            dplus=firstDimCat,
            dminus=firstDimCat,
            hardpct=movingAvg
        )
        self.retId = dict(
            loss=0,
            mask=None,
            dplus=None,
            dminus=None,
            hardpct=None,
            count=0
        )

    def _clip_gradients (self) : 
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)

    def _log_lr (self, ret) : 
        lr = self.opt.state_dict()['param_groups'][0]['lr']
        ret['lr'] = lr

    def _block_step (self, block) : 
        self.model.train()
        ret = self.model(**block)
        return ret

    def training_step(self, batch) :
        self.opt.zero_grad()
        rets = map(self._block_step, batch)
        ret = reduce(
            partial(dictAccumulate, self.combiners), 
            rets, 
            self.retId
        )
        ret['loss'].backward()
        self._clip_gradients()
        self.opt.step()
        self._log_lr(ret)
        tensorApply(
            ret, 
            lambda t : t.item(), 
            lambda t : t.nelement() == 1
        )
        return ret

    def init_validation(self):
        return self.retId

    def validation_step(self, batch, running_data) : 
        self.model.eval()
        with torch.no_grad():
            rets = map(self._block_step, batch)
            ret = reduce(
                partial(dictAccumulate, self.combiners), 
                rets, 
                self.retId
            )
            ret = dictAccumulate(self.combiners, running_data, ret)
            tensorApply(
                ret, 
                lambda t : t.item(), 
                lambda t : t.nelement() == 1
            )
            return ret

def nodeOverlapData(opts) : 
    dir = opts.otherdata
    N   = opts.n_otherdata
    files = [_ for _ in allfiles(dir) if _.endswith('svg')]
    svgs = rng.choices(files, k=N)
    data = [SVGData(_) for _ in svgs]
    return data

def addCallbacks (trainer, model, data, opts) : 
    modelParams = [n for n, _ in model.named_children()]
    keys = ["loss", "hardpct", "lr"]
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
        HardTripletCallback(
            env=opts.name + "_hard_triplet",
            win="hard_triplets",
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
        GradCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_gradients"
        )
    )
    trainer.add_callback(
        InitDistanceCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_init_distance"
        )
    )
    trainer.add_callback(
        NormCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_norms"
        )
    )
    trainer.add_callback(
        TreeScoresCallback(
            model, 
            valData,
            opts,
            env=opts.name + "_treeScores"
        )
    )
    trainer.add_callback(
        HierarchyVisCallback(
            model,
            valData,
            opts,
            env=opts.name + "_hierarchy"
        )
    )
    trainer.add_callback(
        NodeOverlapCallback(
            model, 
            nodeOverlapData(opts),
            opts, 
            env=opts.name + "_no"
        )
    )
    trainer.add_callback(
        AABBVis(
            frequency=opts.frequency,
            env=opts.name + "_vis",
            win='aabb'
        )
    )
    trainer.add_callback(
        OBBVis(
            frequency=opts.frequency,
            env=opts.name + "_vis",
            win='obb'
        )
    )
    trainer.add_callback(
        DistanceHistogramCallback(
            frequency=opts.frequency,
            env=opts.name + "_distance"
        )
    )
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

def buildData (opts) : 
    dataDirs = getDataSplits(opts)
    trainData = TripletDataset(dataDirs[0])
    valData   = TripletDataset(dataDirs[1], trainData.ids)
    testData  = TripletDataset(dataDirs[2], trainData.ids)
    SamplerCls = globals()[opts.samplercls]
    trainDataLoader = TripletDataLoader(
        opts=opts, 
        sampler=SamplerCls(
            trainData, 
            opts.train_epoch_length,
            opts,
            transform=getGraphicAugmentation(opts)
        )
    )
    valDataLoader = TripletDataLoader(
        opts=opts, 
        sampler=SamplerCls(
            valData,
            opts.val_epoch_length,
            opts,
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

def ablations (model, ts, opts) : 
    m1 = buildModel(opts)
    m2 = buildModel(opts)
    m3 = buildModel(opts)

    m1.bbox.apply(getInitializer(opts))
    m2.crop.apply(getInitializer(opts))
    m3.roi.apply(getInitializer(opts))

    ts1_dd = list(map(m1.greedyTree, ts))
    ts1_cg = list(map(m1.containmentGuidedTree, ts))

    ts2_dd = list(map(m2.greedyTree, ts))
    ts2_cg = list(map(m2.containmentGuidedTree, ts))

    ts3_dd = list(map(m3.greedyTree, ts))
    ts3_cg = list(map(m3.containmentGuidedTree, ts))

    dd1 = scores2df(ts, ts1_dd, "- BBox (DD)")
    cg1 = scores2df(ts, ts1_cg, "- BBox (CG)")

    dd2 = scores2df(ts, ts2_dd, "- Crop (DD)")
    cg2 = scores2df(ts, ts2_cg, "- Crop (CG)")

    dd3 = scores2df(ts, ts3_dd, "- RoI (DD)")
    cg3 = scores2df(ts, ts3_cg, "- RoI (CG)")

    combined = pd.concat([dd1, cg1, dd2, cg2, dd3, cg3])
    combined.to_csv(osp.join(opts.checkpoints_dir, opts.name, 'ablations.csv'))

def test (opts) : 
    trainData, _, _, _, testData = buildData(opts)
    graphics = [repr(t.doc) for t in trainData]
    model = buildModel(opts)
    ts1 = [_ for _ in testData if repr(_.doc) not in graphics]
    ablations(model, ts1, opts)
    ts2 = list(map(model.greedyTree, ts1))
    ts3 = list(map(model.containmentGuidedTree, ts1))
    exprDir = osp.join(opts.checkpoints_dir, opts.name)
    logFile = osp.join(exprDir, f'{opts.name}.csv')
    dd = scores2df(ts1, ts2, "Ours-DD") 
    cg = scores2df(ts1, ts3, "Ours-CG")
    combined = pd.concat([dd, cg])
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

