from vectorrvnn.interfaces.helpers import *
import os
import os.path as osp
import importlib

# dynamically import all neural network classes
NN_CLASSES = dict() 
directory = osp.join(osp.dirname(osp.realpath(__file__)), 'models')
lst = [f.rstrip('.py') for f in os.listdir(directory) if f.endswith('.py')]
for item in lst:
    module = importlib.import_module(f'models.{item}')
    NN_CLASSES.update({k: v for k, v in module.__dict__.items() if not k.startswith('_')})

def addCallbacks (trainer, model, data, opts) : 
    addGenericCallbacks(trainer, model, data, opts)
    trainer.add_callback(
        TripletVisCallback(
            env=opts.name + "_vis", 
            win="samples", 
            frequency=opts.frequency
        )
    )
    trainer.add_callback(
        HardTripletCallback(
            env=opts.name + "_hard_triplet",
            win="hard_triplets",
            frequency=opts.frequency
        )
    )
    trainer.add_callback(
        AABBVis(
            frequency=opts.frequency,
            env=opts.name + "_vis",
            win='aabb'
        )
    )

if __name__ == "__main__" : 
    opts = Options().parse()
    setSeed(opts)
    # inject model class 
    opts = dict(opts._asdict())
    opts['modelcls'] = NN_CLASSES[opts['modelcls']]
    Option = namedtuple('Option', [k for k in opts]) 
    opts = Option(*[v for _, v in opts.items()]) 
    # start training
    train(opts, addCallbacks)

