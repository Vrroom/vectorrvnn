from vectorrvnn.interfaces.helpers import *

def addCallbacks (trainer, model, data, opts): 
    keys = ["loss"]
    _, valData, trainDataLoader, _, _ = data
    trainer.add_callback(SchedulerCallback(trainer.interface.sched))
    checkpointer = ttools.Checkpointer(
        osp.join(opts.checkpoints_dir, opts.name),
        model
    )
    trainer.add_callback(CheckpointingCallback(checkpointer))
    trainer.add_callback(
        ProgressBarCallback(keys=keys, val_keys=keys)
    )
    trainer.add_callback(
        LRCallBack(
            trainer.interface.opt,
            env=opts.name + "_lr"
        )
    )
    trainer.add_callback(
        VisdomLoggingCallback(
            keys=keys, 
            val_keys=keys, 
            env=opts.name + "_training_plots", 
            frequency=opts.frequency,
        )
    )
    trainer.add_callback(
        HierarchyVisCallback(
            model,
            valData,
            opts,
            env=opts.name + "_hierarchy"
        )
    )
    trainer.add_callback(
        TreeScoresCallback(
            model, 
            valData,
            opts,
            env=opts.name + "_tree_scores"
        )
    )
    # trainer.add_callback(
    #     NodeOverlapCallback(
    #         model, 
    #         nodeOverlapData(opts),
    #         opts, 
    #         env=opts.name + "_no"
    #     )
    # )
    # trainer.add_callback(
    #     CheckpointingBestNCallback(checkpointer, key='fmi')
    # )
    trainer.add_callback(
        GradCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_gradients"
        )
    )
    trainer.add_callback(
        InitDistanceCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_init_distance"
        )
    )
    trainer.add_callback(
        NormCallback(
            model,
            frequency=opts.frequency,
            env=opts.name + "_norms"
        )
    )

if __name__ == "__main__" : 
    opts = Options().parse()
    setSeed(opts)
    if opts.phase == 'train' : 
        train(opts, addCallbacks)
    elif opts.phase == 'test' : 
        test(opts)


