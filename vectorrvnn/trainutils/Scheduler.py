from torch.optim.lr_scheduler import *

def getScheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; 
        needs to be a subclass of BaseOptions. opt.lr_policy is 
        the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 \
                    - max(0, epoch + 1 - opt.n_epochs) \
                    / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = StepLR(
            optimizer, 
            step_size=opt.lr_decay_iters, 
            gamma=0.1
        )
    elif opt.lr_policy == 'plateau':
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.2, 
            threshold=0.01, 
            patience=5
        )
    elif opt.lr_policy == 'cosine':
        scheduler = CosineAnnealingLR(
            optimizer, 
            T_max=opt.n_epochs, 
            eta_min=0
        )
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', 
            opt.lr_policy
        )
    return scheduler
