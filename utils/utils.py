import torch


def clip_gradient(optimizer, grad_clip):
    """Clip gradients to prevent exploding gradients during training."""
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    """Decay the learning rate by decay_rate every decay_epoch epochs."""
    decay = decay_rate ** (epoch // decay_epoch)
    lr = init_lr * decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
