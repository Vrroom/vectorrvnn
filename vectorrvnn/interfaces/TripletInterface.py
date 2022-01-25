from vectorrvnn.interfaces.helpers import *

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
    trainer.add_callback(
        OBBVis(
            frequency=opts.frequency,
            env=opts.name + "_vis",
            win='obb'
        )
    )

if __name__ == "__main__" : 
    opts = Options().parse()
    setSeed(opts)
    if opts.phase == 'train' : 
        train(opts, addCallbacks)
    elif opts.phase == 'test' : 
        modeltest(opts)

