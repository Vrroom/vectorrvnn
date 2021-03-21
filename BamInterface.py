from torchvision.datasets import ImageFolder
import os
import os.path as osp
from torchvision.models import *
from torchvision import transforms as T
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
from Callbacks import *
from torch.optim.lr_scheduler import MultiStepLR
from Scheduler import *
import ttools 
import ttools.interfaces
from PIL import ImageFile
import warnings
from torch.utils import data

ImageFile.LOAD_TRUNCATED_IMAGES = True

LOG = ttools.get_logger(__name__)

class BamInterface (ttools.ModelInterface) : 

    def __init__(self, model, dataset, lr=0.1, cuda=True, max_grad_norm=10):
        super(BamInterface, self).__init__()
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.device = "cpu"
        self.epoch = 0
        self.cuda = cuda
        self.dataset = dataset
        if cuda:
            self.device = "cuda"
        self.model.to(self.device)
        self.opt = optim.SGD(self.model.parameters(), lr=lr)
        milestones = list(range(1,100))
        self.sched = MultiStepLR(self.opt, milestones, gamma=0.93, verbose=True)
        self.loss = nn.CrossEntropyLoss()

    def training_step(self, batch) :
        self.model.train()
        ims = batch[0].cuda()
        labels = batch[1].cuda()
        result = self.model(ims)
        loss = self.loss(result, labels)
        accuracy = (torch.argmax(result, dim=1) == labels).sum() / labels.shape[0]
        ret = {}
        # optimize
        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()
        ret['loss'] = loss.item()
        ret['accuracy'] = accuracy.item()
        ret["conv-first-layer-kernel"] = self.model.features[0].weight
        return ret

    def init_validation (self) : 
        return {"loss": 0, "accuracy": 0, "count": 0}

    def validation_step(self, batch, running_data) : 
        self.model.eval()
        with torch.no_grad():
            ims = batch[0].cuda()
            labels = batch[1].cuda()
            result = self.model(ims)
            loss = self.loss(result, labels)
            accuracy = (torch.argmax(result, dim=1) == labels).sum() / labels.shape[0]
            n = labels.numel()
            count = running_data['count']
            cumLoss = (running_data["loss"] * count + loss * n) / (count + n)
            cumAccuracy = (running_data["accuracy"] * count + accuracy * n) / (count + n)
        ret = {
            "loss" : cumLoss.item(),
            "count": count + n, 
            "accuracy": cumAccuracy.item()
        }
        return ret

def train (name) : 
    transform = T.Compose([
        T.RandomCrop(64, pad_if_needed=True),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = ImageFolder(
        '/misc/extra/data/sumitc/bam', 
        transform=transform,
        is_valid_file=lambda path : osp.getsize(path) > 10000
    )
    print(dataset.classes)
    n = len(dataset)
    trainN = int(0.8*n)
    valN = n - trainN
    trainData, valData, _ = data.random_split(dataset, [trainN, valN, 0])
    dataLoader = data.DataLoader(
        trainData, 
        batch_size=64, 
        num_workers=6,
        shuffle=True,
        pin_memory=True
    )
    val_dataloader = data.DataLoader(
        valData,
        batch_size=128,
        num_workers=6,
        pin_memory=True
    )
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT = os.path.join(BASE_DIR, "results", name)
    model = alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(4096, 20)
    checkpointer = ttools.Checkpointer(OUTPUT, model)
    interface = BamInterface(model, trainData)
    trainer = ttools.Trainer(interface)
    port = 8097
    keys = ["loss", "accuracy"]
    trainer.add_callback(ttools.callbacks.CheckpointingCallback(checkpointer))
    trainer.add_callback(ttools.callbacks.ProgressBarCallback(keys=keys, val_keys=keys))
    trainer.add_callback(BamImageCallback(env=name + "_vis", win="samples", port=port, frequency=10))
    trainer.add_callback(ttools.callbacks.VisdomLoggingCallback(
        keys=keys, val_keys=keys, env=name + "_training_plots", port=port, frequency=100))
    trainer.add_callback(SchedulerCallback(interface.sched))
    trainer.add_callback(KernelCallback("conv-first-layer-kernel", win="kernel", env=name + "_kernel", port=port, frequency=100))
    # Start training
    trainer.train(dataLoader, num_epochs=400, val_dataloader=val_dataloader)

if __name__ == "__main__" : 
    import sys
    train(sys.argv[1])
