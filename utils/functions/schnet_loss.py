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

def tradeoff_loss_fn(rho, property_names):
    def loss_fn(batch, result):
        loss = 0
        for prop in rho.keys():
            diff = (batch[property_names[prop]] - result[property_names[prop]])**2
            err_sq = rho[prop] * torch.mean(diff)
            loss += err_sq
        return loss
    return loss_fn
