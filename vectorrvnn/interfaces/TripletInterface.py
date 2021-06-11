import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import random
from vectorrvnn.utils import *
from vectorrvnn.data import *
from vectorrvnn.trainutils import *
from vectorrvnn.network import *
import os
import json
import ttools 
import ttools.interfaces
from ttools.callbacks import *
from ttools.modules import networks
from copy import deepcopy

LOG = ttools.get_logger(__name__)

class TripletInterface (ttools.ModelInterface) : 

    def __init__(self, opts, model, dataset, val_dataset, 
            cuda=True, max_grad_norm=10):
        super(TripletInterface, self).__init__()
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.device = "cpu"
        self.epoch = 0
        self.cuda = cuda
        self.dataset = dataset
        self.val_dataset = val_dataset
        if cuda:
            self.device = "cuda"
        self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=opts.lr)
        self.sched = getScheduler(self.opt, opts)
        self.init = deepcopy(self.model.state_dict())

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
        
    def forward (self, batch) : 
        im             = batch['im'].cuda()
        
        # refCrop        = batch['refCrop'].cuda()
        refWhole       = batch['refWhole'].cuda()
        refPositions   = batch['refPositions'].cuda()
        
        # plusCrop       = batch['plusCrop'].cuda()
        plusWhole      = batch['plusWhole'].cuda()
        plusPositions  = batch['plusPositions'].cuda()
        
        # minusCrop      = batch['minusCrop'].cuda()
        minusWhole     = batch['minusWhole'].cuda()
        minusPositions = batch['minusPositions'].cuda()
        
        refPlus        = batch['refPlus'].cuda()
        refMinus       = batch['refMinus'].cuda()
        
        return self.model(
            im, 
            refWhole, refWhole, refPositions, 
            plusWhole, plusWhole, plusPositions, 
            minusWhole, minusWhole, minusPositions, 
            refPlus, refMinus
        )

    def training_step(self, batch) :
        self.model.train()
        result = self.forward(batch)
        dplus2 = result['dplus_']
        dratio = result['dratio']
        mask = result['mask']
        loss = (dplus2.sum() / (dplus2.shape[0] + 1e-6))
        ret = {}
        # optimize
        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()
        ret['mask'] = mask
        ret["loss"] = loss.item()
        ret["conv-first-layer-kernel"] = self.model.conv.conv1.weight
        ret['dratio'] = dratio
        ret['hardRatio'] = result['hardRatio'].item()
        self.logParameterNorms(ret)
        self.logGradients(ret)
        return ret

    def init_validation(self):
        return {"count1": 0, "count2": 0, "loss": 0, "dratio": None, "ratio": None, "hardRatio": 0, "mask": None}

    def validation_step(self, batch, running_data) : 
        self.model.eval()
        with torch.no_grad():
            ratio = batch['refMinus'] / batch['refPlus']
            result = self.forward(batch)
            dratio = result['dratio']
            dplus2 = result['dplus_']
            mask = result['mask']
            hardRatio = result['hardRatio'].item()
            loss = dplus2.mean().item()
            n = ratio.numel()
            n_ = dplus2.numel()
            count1 = running_data['count1']
            count2 = running_data['count2']
            cumLoss = (running_data["loss"] * count1 + loss * n_) / (count1 + n_)
            hardRatio_ = (running_data["hardRatio"] * count2 + hardRatio * n) / (count2 + n)
            dratio_ = dratio if running_data['dratio'] is None else torch.cat([running_data['dratio'], dratio])
            ratio_ = ratio if running_data['ratio'] is None else torch.cat([running_data['ratio'], ratio])
        ret = {
            "loss" : cumLoss,
            "count1": count1 + n_, 
            "count2": count2 + n, 
            "dratio": dratio_,
            "ratio": ratio_,
            "hardRatio": hardRatio_,
            'mask': mask
        }
        return ret

def addCallbacks (trainer, model, opts) : 
    modelParams = [n for n, _ in model.named_children()]
    gradNorms = [f'{n}_grad' for n in modelParams]
    weightDecay = [f'{n}_wd' for n in modelParams]
    keys = ["loss", "hardRatio", *gradNorms, *weightDecay] 
    val_keys=keys[:2]
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
            val_keys=val_keys
        )
    )
    trainer.add_callback(
        ImageCallback(
            env=opts.name + "_vis", 
            win="samples", 
            frequency=opts.frequency
        )
    )
    trainer.add_callback(
        VisdomLoggingCallback(
            keys=keys, 
            val_keys=val_keys, 
            env=opts.name + "_training_plots", 
            frequency=opts.frequency
        )
    )
    trainer.add_callback(
        SchedulerCallback(trainer.interface.sched)
    )
    trainer.add_callback(
        KernelCallback(
            "conv-first-layer-kernel", 
            win="kernel", 
            env=opts.name + "_kernel", 
            frequency=opts.frequency
        )
    )

def buildModel (opts) : 
    # Load pretrained path module
    model = TripletNet(opts).float()
    if opts.load_ckpt is not None : 
        initPath = osp.join(
            opts.checkpoints_dir, 
            opts.load_ckpt
        )
        state_dict = torch.load(
            osp.join(
                initPath, 
                'training_end.pth'
            )
        )
        model.load_state_dict(state_dict['model'])
    return model

def buildData (opts) : 
    trainData = TripletDataset(osp.join(opts.dataroot, 'Train'))
    valData = TripletDataset(osp.join(opts.dataroot, 'Val'))
    trainDataLoader = TripletDataLoader(
        opts=opts, 
        sampler=TripletSampler(
            trainData, 
            opts.train_epoch_length
        )
    )
    valDataLoader = TripletDataLoader(
        opts=opts, 
        sampler=TripletSampler(
            valData,
            opts.val_epoch_length,
            val=True
        )
    )
    return trainData, valData, trainDataLoader, valDataLoader

def train () : 
    opts = Options().parse()
    trainData, valData, trainDataLoader, valDataLoader = buildData(opts)
    model = buildModel(opts) 
    interface = TripletInterface(opts, model, trainData, valData)
    trainer = ttools.Trainer(interface)
    addCallbacks(trainer, model, opts)
    # Start training
    trainer.train(
        trainDataLoader, 
        num_epochs=opts.n_epochs, 
        val_dataloader=valDataLoader
    )

if __name__ == "__main__" : 
    train()

