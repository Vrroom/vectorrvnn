import torch
from torch import nn
import torchvision.models as models
import torch.nn.functional as F

# TODO : Explore finetuning possibilies.
class RasterEncoder (nn.Module) :
    """
    Take the vector-graphic to the same
    latent space as the RvNN root code.
    """
    def __init__ (self, feature_size) :
        super(RasterEncoder, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        for param in self.resnet18.parameters():
            param.requires_grad = False
        self.mlp = nn.Linear(1000, feature_size)

    def forward (self, image) :
        features = self.resnet18(image)
        return self.mlp(features)


