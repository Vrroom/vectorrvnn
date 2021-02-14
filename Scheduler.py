from torch.optim.lr_scheduler import _LRScheduler

class DecayToZeroLR(_LRScheduler) :

    def __init__ (self, optimizer, start_epoch, end_epoch, last_epoch=-1, verbose=True) :
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        super(DecayToZeroLR, self).__init__(optimizer, last_epoch, verbose)

    def compute (self, lr) : 
        epoch = self.last_epoch + 1
        if epoch <= self.start_epoch : 
            return lr
        else : 
            return lr * (self.end_epoch - epoch) / (self.end_epoch - self.start_epoch)

    def get_lr (self) : 
        return [self.compute(lr) for lr in self.base_lrs]
        
        

