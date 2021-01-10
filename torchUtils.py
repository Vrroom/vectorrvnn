from torchvision import transforms as T
import numpy as np
import torchvision.models as models
import torch
from torch import nn
import torch.nn.functional as F

class MyBN (nn.BatchNorm2d) : 

    def forward (self, x) : 
        thing = super(MyBN, self).forward(x)
        # import pdb
        # pdb.set_trace()
        self.mean_diff = torch.linalg.norm(self.running_mean - x.mean(dim=(0, 2, 3)))
        self.var_diff = torch.linalg.norm(self.running_var - x.var(dim=(0, 2, 3)))
        # print(self.mean_diff, self.var_diff)
        return thing

def convLayer (in_channel, out_channel, kernel_size, stride) :
    return nn.Sequential(
        nn.Conv2d(in_channel, out_channel, kernel_size, stride),
        # MyBN(out_channel),
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

if __name__ == "__main__" : 
    import matplotlib.pyplot as plt
    import matplotlib.image as image
    resnet  = models.resnet18(pretrained=True).cuda()
    import pdb
    pdb.set_trace()
    pass
    im = torch.randn(10, 3, 100, 100).cuda()
    resnet(im)
    # im = image.imread('elephant.jpg', 'jpg').copy()
    # im = im / 255 
    # elephant = imageForResnet(im, True).unsqueeze(0)
    # fig, ax = plt.subplots(1)
    # # ax.imshow(np.transpose(elephant.cpu().squeeze().numpy(), (1, 2, 0)))
    # # fig.savefig('elephant')
    # # print(elephant.min(), elephant.max(), elephant.shape)
    # logits = resnet(elephant).squeeze()
    # pred = torch.argmax(resnet(elephant).squeeze())
    print(pred)
