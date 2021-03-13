from torchvision import transforms as T
import numpy as np
import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F

class MyBN (nn.BatchNorm2d) : 

    def forward (self, x) : 
        thing = super(MyBN, self).forward(x)
        self.mean_diff = torch.linalg.norm(self.running_mean - x.mean(dim=(0, 2, 3)))
        self.var_diff = torch.linalg.norm(self.running_var - x.var(dim=(0, 2, 3)))
        return thing

def convLayer (in_channel, out_channel, kernel_size, stride) :
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride),
        nn.ReLU()
    )

def imageForResnet (img, cuda=False) :
    img = torch.from_numpy(img).float()
    img = img.permute(2, 0, 1)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    img = normalizer(img)
    if cuda : 
        img = img.cuda()
    return img
