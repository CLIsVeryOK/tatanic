import torch.optim as optim


def BuildLRScheduler(optimizer, step, factor, last_epoch=-1):
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                  last_epoch=last_epoch,
                                                  milestones=step,
                                                  gamma=factor)
    return lr_scheduler
