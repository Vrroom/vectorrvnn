import torch
from torch import nn
from vectorrvnn.utils import * 
from vectorrvnn.trainutils import *
from torchvision.ops import roi_align
from torchvision.models import *

def _setUnitStride (m) : 
    if isinstance(m, nn.Conv2d) : 
        m.stride = (1, 1)

class RoIAlignNet (nn.Module) : 

    def __init__ (self, opts)  :
        super(RoIAlignNet, self).__init__()
        self.opts = opts
        self.resnet = convBackbone(opts)
        # make the modifications to resnet based on mask rcnn paper.
        self.resnet.layer4.apply(_setUnitStride)

    def forward (self, ims, boxes) : 
        """
        ims - [B, C, H, W]
        boxes - [B, 4]
        """
        B, *_ = ims.shape

        # concatenate the batch index as per
        # https://pytorch.org/vision/stable/ops.html#torchvision.ops.roi_align
        idx = torch.arange(B).unsqueeze(1).to(self.opts.device)
        boxes = torch.cat((idx, boxes), dim=1)

        # Do the forward pass till Layer 3
        x = self.resnet.conv1(ims)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)

        # Make the RoIAlign computation
        _, _, H, W = x.shape
        assert H == W
        x = roi_align(x, boxes, output_size=(7, 7), spatial_scale=H)

        # Make the rest of the forward pass
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)

        return x

