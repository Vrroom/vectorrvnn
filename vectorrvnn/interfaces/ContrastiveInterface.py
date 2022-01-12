from .helpers import *

if __name__ == "__main__" : 
    opts = Options().parse()
    setSeed(opts)
    if opts.phase == 'train' : 
        train(opts, addGenericCallbacks)
    elif opts.phase == 'test' : 
        test(opts)

