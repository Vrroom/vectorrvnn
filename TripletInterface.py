import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import random
from listOps import avg
import matplotlib.pyplot as plt
from itertools import product
import os
from Triplet import *
from TripletSVGData import *
from TripletDataset import *
import json
import more_itertools
import math
import ttools 
import ttools.interfaces
from ttools.modules import networks
from dictOps import aggregateDict
import visdom
from torchUtils import *
from Callbacks import *
from Scheduler import *

LOG = ttools.get_logger(__name__)

class TripletInterface (ttools.ModelInterface) : 

    def __init__(self, model, dataset, val_dataset, lr=3e-4, cuda=True, max_grad_norm=10,
                 variational=True):
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
        self.opt = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        milestones = [100, *range(200, 400, 10)]
        self.sched = MultiStepLR(self.opt, milestones, gamma=0.5, verbose=True)

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
        
    def logBNDiff (self, ret) : 
        with torch.no_grad() : 
            bns = list(filter(lambda x : isinstance(x, MyBN), self.model.conv.modules()))
            mean_diff = sum(map(lambda x: x.mean_diff, bns)).item()
            var_diff = sum(map(lambda x: x.var_diff, bns)).item()
            ret['mean_diff'] = mean_diff
            ret['var_diff'] = var_diff

    @lru_cache
    def getNodeEmbedding (self, tId, node, dataset) : 
        pt = dataset.getNodeInput(tId, node)
        im    = pt['im'].cuda()
        crop  = pt['crop'].cuda()
        whole = pt['whole'].cuda()
        return self.model.embedding(im, crop, whole)

    def parentAndChildrenEmbeddings(self, tId, p, dataset) :
        t = dataset.svgDatas[tId]
        children = list(t.neighbors(p))
        pEmbedding = self.getNodeEmbedding(tId, p, dataset)
        childrenEmbedding = [self.getNodeEmbedding(tId, c, dataset) for c in children]
        return (pEmbedding, childrenEmbedding)

    def parent2ChildrenAvgDistance (self, tId, p, dataset) : 
        pEmbedding, childrenEmbedding = self.parentAndChildrenEmbeddings(tId, p, dataset) 
        val = avg([torch.linalg.norm(cEmbedding - pEmbedding) for cEmbedding in childrenEmbedding])
        return val

    def parent2ChildrenCentroidDistance (self, tId, p, dataset) : 
        pEmbedding, childrenEmbedding = self.parentAndChildrenEmbeddings(tId, p, dataset) 
        centroid = avg(childrenEmbedding)
        return torch.linalg.norm(pEmbedding - centroid)

    def parent2ChildrenSubspaceProjection (self, tId, p, dataset) : 
        pEmbedding, childrenEmbedding = self.parentAndChildrenEmbeddings(tId, p, dataset) 
        A = torch.stack(childrenEmbedding).squeeze()
        x = A.t() @ torch.inverse(A @ A.t()) @ A @ pEmbedding.t()
        x = x.t()
        return torch.linalg.norm(pEmbedding - x)
    
    def sampleTP (self, dataset) : 
        try :
            tId = random.randint(0, len(dataset.svgDatas) - 1)
            t = dataset.svgDatas[tId]
            p   = random.sample([n for n in t.nodes if t.out_degree(n) > 0], k = 1).pop()
        except Exception : 
            return self.sampleTP(dataset)
        return tId, p

    def estimate (self, metric, dataset) : 
        estimates = []
        for i in range(10) : 
            tId, p = self.sampleTP(dataset)
            estimates.append(metric(tId, p, dataset))
        return avg(estimates)

    def fillEstimates (self, ret, dataset) : 
        with torch.no_grad(): 
            e1 = self.estimate(self.parent2ChildrenAvgDistance, dataset)
            e2 = self.estimate(self.parent2ChildrenCentroidDistance, dataset)
            # e3 = self.estimate(self.parent2ChildrenSubspaceProjection, dataset)
            ret['avg-distance'] = e1.item()
            ret['centroid-distance'] = e2.item()
            # ret['subspace-projection'] = e3.item()

    def forward (self, batch) : 
        im = batch['im'].cuda()
        refCrop = batch['refCrop'].cuda()
        refWhole = batch['refWhole'].cuda()
        plusCrop = batch['plusCrop'].cuda()
        plusWhole = batch['plusWhole'].cuda()
        minusCrop = batch['minusCrop'].cuda()
        minusWhole = batch['minusWhole'].cuda()
        refPlus = batch['refPlus'].cuda()
        refMinus = batch['refMinus'].cuda()
        return self.model(
            im, 
            refCrop, refWhole, 
            plusCrop, plusWhole, 
            minusCrop, minusWhole, 
            refPlus, refMinus
        )

    def training_step(self, batch) :
        self.model.train()
        result = self.forward(batch)
        dplus2 = result['dplus_']
        dratio = result['dratio']
        loss = dplus2.sum() / (dplus2.shape[0] + 1e-6)
        ret = {}
        # optimize
        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()
        ret["loss"] = loss.item()
        ret["conv-first-layer-kernel"] = self.model.conv[0][0].weight
        ret['dratio'] = dratio
        ret['hardRatio'] = result['hardRatio'].item()
        self.logParameterNorms(ret)
        self.logGradients(ret)
        self.fillEstimates(ret, self.dataset)
        return ret

    def init_validation(self):
        return {"count1": 0, "count2": 0, "loss": 0, "dratio": None, "ratio": None, "hardRatio": 0}

    def validation_step(self, batch, running_data) : 
        self.model.eval()
        with torch.no_grad():
            ratio = batch['refMinus'] / batch['refPlus']
            result = self.forward(batch)
            dratio = result['dratio']
            dplus2 = result['dplus_']
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
            "hardRatio": hardRatio_
        }
        self.fillEstimates(ret, self.val_dataset)
        return ret

def evalVal(model) : 
    model.eval()
    testData = TripletSVGDataSet('cv64.pkl').svgDatas
    testData = [t for t in testData if t.nPaths < 50]
    scoreFn = lambda t, t_ : ted(t, t_) / (t.number_of_nodes() + t_.number_of_nodes())
    testData = list(map(treeify, testData))
    inferredTrees = [model.greedyTree(t) for t in tqdm(testData)]
    scores = [scoreFn(t, t_) for t, t_ in tqdm(zip(testData, inferredTrees), total=len(testData))]
    print(np.mean(scores))

def train (name) : 
    with open('Configs/config.json') as fd : 
        config = json.load(fd)
    trainData = TripletSVGDataSet('train4channel.pkl', whiteBackgroundTransform)
    dataLoader = torch.utils.data.DataLoader(
        trainData, 
        batch_size=32, 
        sampler=TripletSampler(trainData.svgDatas, 10000),
        pin_memory=True,
        collate_fn=lambda x : aggregateDict(x, torch.stack)
    )
    valData = TripletSVGDataSet('cv4channel.pkl', whiteBackgroundTransform)
    valDataLoader = torch.utils.data.DataLoader(
        valData, 
        batch_size=128,
        sampler=TripletSampler(valData.svgDatas, 1000, val=True),
        pin_memory=True,
        collate_fn=lambda x : aggregateDict(x, torch.stack)
    )
    # Load pretrained path module
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MERGE_OUTPUT = os.path.join(BASE_DIR, "results", name)
    # Initiate main model.
    model = TripletNet(dict(hidden_size=100)).float()
    checkpointer = ttools.Checkpointer(MERGE_OUTPUT, model)
    interface = TripletInterface(model, trainData, valData)
    trainer = ttools.Trainer(interface)
    port = 8097
    named_children = [n for n, _ in model.named_children()]
    named_grad = [f'{n}_grad' for n in named_children]
    named_wd = [f'{n}_wd' for n in named_children]
    keys = ["loss", "hardRatio", "avg-distance", "centroid-distance", *named_grad, *named_wd]
    val_keys=keys[:3]
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=val_keys, val_keys=val_keys))
    trainer.add_callback(ImageCallback(env=name + "_vis", win="samples", port=port, frequency=100))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=val_keys, env=name + "_training_plots", port=port, frequency=100))
    trainer.add_callback(SchedulerCallback(interface.sched))
    # Start training
    trainer.train(dataLoader, num_epochs=200, val_dataloader=valDataLoader)
    evalVal(model)

if __name__ == "__main__" : 
    import sys
    train(sys.argv[1])

