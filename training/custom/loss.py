import torch

def default_loss():
    def loss(batch, result):
        diff = (batch["fx"] - result["fx"]) ** 2
        err_sq = torch.mean(diff)
        return err_sq
    return loss

def down_loss():
    def loss(batch, result):
        diff = (result["fx"] - batch["fx"])
        idx = torch.where(diff>=0)
        diff[idx] = diff[idx] * 4.
        err = torch.mean(diff ** 2)
        return err
    return loss

def up_loss():
    def loss(batch, result):
        diff = (batch["fx"] - result["fx"])
        idx = torch.where(diff<=0)
        diff[idx] = diff[idx] * 4.
        err = torch.mean(diff ** 2)
        return err
    return loss
