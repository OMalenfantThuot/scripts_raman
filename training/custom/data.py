import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler


def get_data(args):
    function = get_function(args.function)
    x = np.linspace(args.range[0], args.range[1], args.ndata)
    fx = function.value(x)
    return x, fx


def get_function(name):
    if name == "sin":
        from utils.functions.analytic_functions import Sin

        return Sin()
    elif name == "cos":
        from utils.functions.analytic_functions import Cos

        return Cos()
    elif name == "lj" or name == "lennardjones":
        from utils.functions.analytic_functions import LennardJones

        return LennardJones()
    elif name == "quad":
        from utils.functions.analytic_functions import Quadratic

        return Quadratic()
    elif name == "p4":
        from utils.functions.analytic_functions import Polynomial_order4

        return Polynomial_order4()
    else:
        raise NotImplementedError


class CustomDataset(Dataset):
    def __init__(self, x, fx):
        assert len(x) == len(fx), "Data is incompatible."
        self.x = x
        self.fx = fx

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        point = {"x": self.x[idx], "fx": self.fx[idx]}
        return point


def split_data(args, x, fx):
    if len(args.split) != 2:
        raise ValueError("Data splitting invalid.")
    if np.sum(args.split) > args.ndata:
        raise ValueError("Splits are too large for the data.")

    full_idx = np.linspace(0, args.ndata - 1, args.ndata).astype(int)
    np.random.shuffle(full_idx)

    train_idx, val_idx = (
        full_idx[: args.split[0]],
        full_idx[args.split[0] : args.split[0] + args.split[1]],
    )
    train_data = CustomDataset(x[train_idx], fx[train_idx])
    val_data = CustomDataset(x[val_idx], fx[val_idx])
    return train_data, val_data


def get_loaders(train_data, val_data):
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=8, shuffle=False)
    return train_loader, val_loader
