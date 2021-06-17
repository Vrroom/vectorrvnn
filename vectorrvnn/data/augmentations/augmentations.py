from .transform import *
from .composition import *

simple = OneOf([
    NoFill(), 
    Rotate()
], p=0.5)

def getGraphicAugmentation (opts) : 
    if opts.augmentation == 'none' : 
        return lambda *args:  args[0]
    else : 
        return globals()[opts.augmentation]

