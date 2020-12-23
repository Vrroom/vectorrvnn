from torchvision import transforms as T
import numpy as np
import torchvision.models as models
import torch

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
    resnet  = models.resnet50(pretrained=True).cuda()
    im = image.imread('elephant.jpg', 'jpg').copy()
    im = im / 255 
    elephant = imageForResnet(im, True).unsqueeze(0)
    fig, ax = plt.subplots(1)
    # ax.imshow(np.transpose(elephant.cpu().squeeze().numpy(), (1, 2, 0)))
    # fig.savefig('elephant')
    # print(elephant.min(), elephant.max(), elephant.shape)
    logits = resnet(elephant).squeeze()
    pred = torch.argmax(resnet(elephant).squeeze())
    print(pred)
