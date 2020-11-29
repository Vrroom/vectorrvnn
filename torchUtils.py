from torchvision import transforms as T
import torch

def imageForResnet (image, cuda=False) :
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1)
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    image = normalizer(image)
    if cuda : 
        image = image.cuda()
    return image
