import os
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

sys.path.append(os.environ["RAMAN"])


def get_data(args):
    function = get_function(args.function)
    x = np.linspace(args.range[0], args.range[1], args.ndata)
    fx = function.value(x)
    return x, fx


def get_function(name):
    if name == "sin":
        from utils.analytic_functions import Sin

        return Sin()
    elif name == "cos":
        from utils.analytic_functions import Cos

        return Cos()
    elif name == "lj" or name == "lennardjones":
        from utils.analytic_functions import LennardJones

        return LennardJones()
    elif name == "quad":
        from utils.analytic_functions import Quadratic

        return Quadratic()
    elif name == "p4":
        from utils.analytic_functions import Polynomial_order4

        return Polynomial_order4()
    else:
        raise NotImplementedError


class CustomDataset(Dataset):
    def __init__(self, x, fx):
        assert len(x) == len(fx), "Data is incompatible."
        self.x = x
        self.fx = fx

    def __len__(self):
        return len(x)

    def __getitem__(self, idx):
        point = {"x": self.x[idx], "fx": self.fx[idx]}


def get_dataset(x, fx):
    return CustomDataset(x, fx)


def split_data(split):
    if len(args.split) != 2:
        raise ValueError("Data splitting invalid.")
    if np.sum(split) > len(x):
        raise ValueError("Splits are too large for the data.")

    full_idx = np.linspace(0, len(x) - 1, len(x)).astype(int)
    np.random.shuffle(full_idx)

    train_idx, val_idx = full_idx[: split[0]], full_idx[split[0] : split[0] + split[1]]
    return train_idx, val_idx


def get_loaders(dataset, train_idx, val_idx):
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset, batch_size=8, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=8, sampler=val_sampler)
    return train_sampler, val_sampler
