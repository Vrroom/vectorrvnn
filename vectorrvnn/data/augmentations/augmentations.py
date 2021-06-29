from .transform import *
from .composition import *

simple = Rotate(p=0.3)

oneof = OneOf([
    Rotate(),
    NoFill()
], p=0.3)

def getGraphicAugmentation (opts) : 
    if opts.augmentation == 'none' : 
        return lambda *args:  args[0]
    else : 
        return globals()[opts.augmentation]

