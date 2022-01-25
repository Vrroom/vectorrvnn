from vectorrvnn.interfaces.helpers import *

def addCallbacks (trainer, model, data, opts) : 
    addGenericCallbacks(trainer, model, data, opts)
    trainer.add_callback(
        VisContrastiveExample(
            opts,
            env=opts.name + "_vis", 
            frequency=opts.frequency
        )
    )

if __name__ == "__main__" : 
    opts = Options().parse()
    setSeed(opts)
    if opts.phase == 'train' : 
        train(opts, addCallbacks)
    elif opts.phase == 'test' : 
        test(opts)

