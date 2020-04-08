import torch

def simple_fn(args):
    def loss(batch, result):
        diff = (batch[args.property] - result[args.property]) ** 2
        err_sq =torch.mean(diff)
        return err_sq
    return loss

def tilted_down(args):
    def loss(batch, result):
        diff = (batch[args.property] - result[args.property])
        idx = torch.where(diff>=0)
        diff[idx] = diff[idx] * 4.
        err = torch.mean(diff ** 2)
        return err
    return loss

def tilted_up(args):
    def loss(batch, result):
        diff = (batch[args.property] - result[args.property])
        idx = torch.where(diff<=0)
        diff[idx] = diff[idx] * 4.
        err = torch.mean(diff ** 2)
        return err
    return loss
