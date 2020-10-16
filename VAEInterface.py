import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
import ttools 
import ttools.interfaces
from ttools.modules import networks

class VAEInterface(ttools.ModelInterface):
    def __init__(self, model, n_epoch, lr=1e-4, cuda=True, max_grad_norm=10,
                 variational=True):
        super(VAEInterface, self).__init__()
        self.max_grad_norm = max_grad_norm
        self.model = model
        self.variational = variational
        self.device = "cpu"
        self.epoch = 0
        if cuda:
            self.device = "cuda"
        self.model.to(self.device)
        self.opt = optim.Adam(self.model.parameters(), lr=lr)
        self.w_kld = self._schedule(0.0, 0.001, n_epoch)

    def _schedule (self, start, stop, n_epoch, n_cycle=5, ratio=1): 
        L = np.ones(n_epoch)
        period = n_epoch/n_cycle
        step = (stop-start)/(period*ratio) # linear schedule
        for c in range(n_cycle):
            v, i = start , 0
            while v <= stop and (int(i+c*period) < n_epoch):
                L[int(i+c*period)] = v
                v += step
                i += 1
        return L
        
    def forward(self, batch):
        batch = batch.to(self.device)
        out, auxdata = self.model(batch)
        return out, auxdata

    def backward(self, batch, fwd_data):
        rendering, aux_data = fwd_data
        batch = batch.to(self.device)
        logvar = aux_data["logvar"]
        mu = aux_data["mu"]
        data_loss = F.mse_loss(rendering, batch)
        ret = {}
        if self.variational:
            kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1)
            kld = kld.mean()
            loss = data_loss + kld*self.w_kld[self.epoch]
            ret["kld"] = kld.item()

            # Weight decay
            reg_loss = 0
            for p in self.model.parameters():
                reg_loss += p.pow(2).sum()

            # loss = loss + 1e-4*reg_loss

            ret["wd"] = reg_loss.item()
        else:
            loss = data_loss
        # optimize
        self.opt.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nrm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if nrm > self.max_grad_norm:
                LOG.warning("Clipping generator gradients. norm = %.3f > %.3f", nrm, self.max_grad_norm)
        self.opt.step()
        ret["loss"] = loss.item()
        ret["data_loss"] = data_loss.item()
        return ret

    def init_validation(self):
        return {"count": 0, "loss": 0}

    def update_validation(self, batch, fwd, running_data):
        with torch.no_grad():
            ref = batch[1].to(self.device)
            loss = F.mse_loss(fwd, ref)
            n = ref.shape[0]
        return {
            "loss": running_data["loss"] + loss.item()*n,
            "count": running_data["count"] + n
        }

    def finalize_validation(self, running_data):
        return {
            "loss": running_data["loss"] / running_data["count"]
        }

