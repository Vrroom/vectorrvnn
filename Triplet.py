import torch
from torch import nn
import torch.nn.functional as F
from torchUtils import * 

def smallConvNet () : 
    return nn.Sequential(
        convLayer(3, 64, 5, 1),
        nn.MaxPool2d(2),
        convLayer(64, 128, 3, 1), 
        nn.MaxPool2d(2),
        convLayer(128, 256, 3, 1), 
        nn.MaxPool2d(2),
        nn.Conv2d(256, 128, 2),
        nn.Flatten()
    )

class TripletNet (nn.Module) :

    def __init__ (self, config) :
        super(TripletNet, self).__init__() 
        self.hidden_size = config['hidden_size']
        self.conv = smallConvNet()
        self.nn = nn.Sequential(
            nn.Linear(256, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 128)
        )

    def forward (self, im, ref, plus, minus) : 
        global_embed = self.conv(im)
        ref_embed = self.conv(ref)
        plus_embed = self.conv(plus)
        minus_embed = self.conv(minus)
        ref_embed = self.nn(torch.cat((global_embed, ref_embed), dim=1))
        plus_embed = self.nn(torch.cat((global_embed, plus_embed), dim=1))
        minus_embed = self.nn(torch.cat((global_embed, minus_embed), dim=1))
        dplus  =  torch.sqrt(torch.sum((plus_embed  - ref_embed) ** 2, dim=1, keepdims=True))
        dminus =  torch.sqrt(torch.sum((minus_embed - ref_embed) ** 2, dim=1, keepdims=True))
        o = F.softmax(torch.cat((dplus, dminus), dim=1), dim=1)
        dplus_ = o[:, 0] ** 2
        return dplus_

if __name__ == "__main__" : 
    m = TripletNet(dict(hidden_size=100))
    im = torch.randn((10, 3, 32, 32))
    ref = torch.randn((10, 3, 32, 32))
    plus = torch.randn((10, 3, 32, 32))
    minus = torch.randn((10, 3, 32, 32))
    o = m.forward(im, ref, plus, minus)
