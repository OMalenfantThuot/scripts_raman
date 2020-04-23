#!/usr/bin/env python
import torch
import numpy as np
from parser import build_parser
from data import get_data, split_data, get_loaders, save_splits
from utils.models import get_model
from trainer import get_trainer


def main(args):
    x, fx = get_data(args)

    device = torch.device("cuda" if args.cuda else "cpu")
    train_data, val_data = split_data(args, x, fx)

    if args.save_splits:
        save_splits(train_data, val_data)
    
    train_loader, val_loader = get_loaders(train_data, val_data)

    model = get_model(args)

    trainer = get_trainer(model, train_loader, val_loader, device, args)
    trainer.train()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
