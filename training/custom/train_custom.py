#!/usr/bin/env python
import torch
from parser import build_parser
from data import get_data, split_data, get_loaders
from utils.models import get_model
from trainer import get_trainer


def main(args):
    x, fx = get_data(args)

    device = torch.device("cuda" if args.cuda else "cpu")
    train_data, val_data = split_data(args, x, fx)

    train_loader, val_loader = get_loaders(train_data, val_data)

    model = get_model(args)

    trainer = get_trainer(model, train_loader, val_loader, device, args)
    trainer.train()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
