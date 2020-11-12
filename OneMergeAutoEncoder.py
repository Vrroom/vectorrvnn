import json
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dictOps import aggregateDict
from Dataset import SVGDataSet
from PathModules import * 
import torch
from torch import nn
import torch.optim as optim
from VAEInterface import VAEInterface
import ttools 
import ttools.interfaces
from ttools.modules import networks
import visdom
from losses import iou
from RvNNModules import MLPMergeEncoder, MLPMergeDecoder

class OneMergeAutoEncoder (nn.Module) : 

    def __init__ (self, pathVAE, config) : 
        super(OneMergeAutoEncoder, self).__init__()
        self.pathVAE = pathVAE
        self.mergeEncoder = MLPMergeEncoder(config)
        self.mergeDecoder = MLPMergeDecoder(config)
    
    def forward (self, x, numNeighbors) : 
        x, _ = self.pathVAE.encode(x)
        x = self.mergeEncoder(x, numNeighbors)
        x = self.mergeDecoder(x)
        x, _ = self.pathVAE.decode(x)
        return x
        
