from .transform import *
from .composition import *

oneof = OneOf([
    Rotate(),
    NoFill()
], p=0.3)

multiaug = Compose([
    OneOf([
        Rotate(),
        NoFill()
    ], p=0.4),
    OneOf([
        StrokeWidthJitter(scaleRange=(0.7, 1.3)),
        OpacityJitter(lowerBound=0.7),
        HSVJitter()
    ], p=0.4),
    GraphicCompose(p=0.4)
])

binarize = RandomBinarize(p=1)

simple = Compose([
    RandomBinarize(p=0.5),
    Rotate(p=0.5),
    NoFill(p=0.5),
])

def getGraphicAugmentation (opts) : 
    if opts.augmentation == 'none' : 
        return lambda *args:  args[0]
    else : 
        return globals()[opts.augmentation]

