#!/usr/bin/env python
import os
import sys
import torch
from torch.optim import Adam
import logging
import torch.nn as nn
import schnetpack as spk
from schnetpack.utils import (
    read_from_json,
    get_dataset,
    get_loaders,
    get_metrics,
    get_model,
    setup_run,
    get_statistics,
    get_divide_by_atoms,
)

from schnetpack.utils.script_utils.settings import get_environment_provider
from schnetpack.utils.script_utils.parsing import build_parser
from schnetpack.utils.script_utils.model import get_output_module

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def main(args):

    # setup
    train_args = setup_run(args)
    device = torch.device("cuda" if args.cuda else "cpu")

    # get dataset
    environment_provider = get_environment_provider(train_args, device=device)
    dataset = get_dataset(train_args, environment_provider=environment_provider)

    # get dataloaders
    split_path = os.path.join(args.modelpath, "split.npz")
    train_loader, val_loader, test_loader = get_loaders(
        args, dataset=dataset, split_path=split_path, logging=logging
    )

    # define metrics
    metrics = get_metrics(train_args)

    # train or evaluate
    if args.mode == "train":

        # get statistics
        atomref = dataset.get_atomref(args.property)
        mean, stddev = get_statistics(
            args=args,
            split_path=split_path,
            train_loader=train_loader,
            atomref=atomref,
            divide_by_atoms=get_divide_by_atoms(args),
            logging=logging,
        )

        # build model
        model = get_model(args, train_loader, mean, stddev, atomref, logging=logging)

        # build trainer
        logging.info("training...")
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)

        # run training
        trainer.train(device, n_epochs=args.n_epochs)
        logging.info("...training done!")

    else:
        raise ("Use the original SchnetPack script instead.")


def get_trainer(args, model, train_loader, val_loader, metrics):
    # setup optimizer
    # filter for trainable parameters (https://github.com/pytorch/pytorch/issues/679)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_params, lr=args.lr)

    # setup hook and logging
    hooks = [spk.train.MaxEpochHook(args.max_epochs)]
    if args.max_steps:
        hooks.append(spk.train.MaxStepHook(max_steps=args.max_steps))

    schedule = spk.train.ReduceLROnPlateauHook(
        optimizer=optimizer,
        patience=args.lr_patience,
        factor=args.lr_decay,
        min_lr=args.lr_min,
        window_length=1,
        stop_after_min=True,
    )
    hooks.append(schedule)

    if args.logger == "csv":
        logger = spk.train.CSVHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)
    elif args.logger == "tensorboard":
        logger = spk.train.TensorboardHook(
            os.path.join(args.modelpath, "log"),
            metrics,
            every_n_epochs=args.log_every_n_epochs,
        )
        hooks.append(logger)

    # setup loss function
    loss_fn = get_loss_fn(args)

    # setup trainer
    trainer = spk.train.Trainer(
        args.modelpath,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_interval=args.checkpoint_interval,
        keep_n_checkpoints=args.keep_n_checkpoints,
        hooks=hooks,
    )
    return trainer


def get_loss_fn(args):
    derivative = spk.utils.get_derivative(args)
    contributions = spk.utils.get_contributions(args)
    stress = spk.utils.get_stress(args)
    if args.loss in ["default", "tilted_up", "tilted_down"]:
        loss = args.loss
    else:
        raise ValueError("The loss argument is not recognized.")
    if loss == "default":
        # simple loss function for training on property only
        if derivative is None and contributions is None and stress is None:
            from utils.functions.schnet_loss import simple_fn

            return simple_fn(args)

        # loss function with tradeoff weights
        if type(args.rho) == float:
            rho = dict(property=args.rho, derivative=1 - args.rho)
        else:
            rho = dict()
            rho["property"] = (
                1.0 if "property" not in args.rho.keys() else args.rho["property"]
            )
            if derivative is not None:
                rho["derivative"] = (
                    1.0
                    if "derivative" not in args.rho.keys()
                    else args.rho["derivative"]
                )
            if contributions is not None:
                rho["contributions"] = (
                    1.0
                    if "contributions" not in args.rho.keys()
                    else args.rho["contributions"]
                )
            if stress is not None:
                rho["stress"] = (
                    1.0 if "stress" not in args.rho.keys() else args.rho["stress"]
                )
            # type cast of rho values
            for key in rho.keys():
                rho[key] = float(rho[key])
            # norm rho values
            norm = sum(rho.values())
            for key in rho.keys():
                rho[key] = rho[key] / norm
        property_names = dict(
            property=args.property,
            derivative=derivative,
            contributions=contributions,
            stress=stress,
        )
        return tradeoff_loss_fn(rho, property_names)
    elif loss == "tilted_down":
        if derivative is None and contributions is None and stress is None:
            from utils.functions.schnet_loss import tilted_down

            return tilted_down(args)
    elif loss == "tilted_up":
        if derivative is None and contributions is None and stress is None:
            from utils.functions.schnet_loss import tilted_up

            return tilted_up(args)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "from_json":
        args = read_from_json(args.json_path)
    main(args)
