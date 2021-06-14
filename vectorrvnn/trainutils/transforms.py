from vectorrvnn.utils import *
import torchvision.transforms as T

def getTransform (opts) : 
    transforms = [
        T.ToTensor(), 
        T.Lambda(lambda t : t.float()),
        T.Normalize(mean=opts.mean, std=opts.std)
    ]
    if opts.input_nc < 4 : 
        transforms.insert(
            1, 
            T.Lambda(
                partial(alphaComposite, module=torch)
            )
        )
    return T.Compose(transforms)
