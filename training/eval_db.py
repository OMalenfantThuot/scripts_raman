#!/usr/bin/env python

import os
import csv
import torch
import argparse
import schnetpack as spk
from schnetpack.utils.script_utils.evaluation import evaluate_dataset


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dbpath", help="Path to the database to evaluate.")
    parser.add_argument("modelpath", help="Path to the model directory.")
    parser.add_argument(
        "--overwrite", help="Remove previous evaluation file", action="store_true"
    )
    parser.add_argument("--cuda", help="Use GPU for evaluation", action="store_true")
    return parser


def main(args):

    eval_file = os.path.join(args.modelpath, "eval.txt")
    if os.path.exists(eval_file):
        if args.overwrite:
            os.remove(eval_file)
        else:
            raise Exception(
                "The evaluation file already exists. Delete it or add the overwrite flag."
            )

    train_args = spk.utils.read_from_json(os.path.join(args.modelpath, "args.json"))
    device = torch.device("cuda" if args.cuda else "cpu")
    environment_provider = spk.utils.script_utils.get_environment_provider(train_args, device)

    dataset = get_dataset(args, train_args, environment_provider)

    loader = spk.data.AtomsLoader(
        dataset, batch_size=train_args.batch_size, num_workers=2, pin_memory=True
    )

    model = spk.utils.load_model(
        os.path.join(args.modelpath, "best_model"), map_location=device
    )

    metrics = spk.utils.get_metrics(train_args)

    if spk.utils.get_derivative(train_args) is None:
        with torch.no_grad():
            header, results = evaluate(train_args, model, loader, device, metrics)
    else:
        header, results = evaluate(train_args, model, loader, device, metrics)

    with open(eval_file, "w") as file:
        wr = csv.writer(file)
        wr.writerow(header)
        wr.writerow(results)


def get_dataset(args, train_args, environment_provider):
    load_only = [train_args.property]
    try:
        if train_args.derivative is not None:
            load_only.append(train_args.derivative)
    except:
        pass

    dataset = spk.AtomsData(
        args.dbpath,
        load_only=load_only,
        collect_triples=train_args.model == "wacsf",
        environment_provider=environment_provider,
    )
    return dataset


def evaluate(train_args, model, loader, device, metrics):
    header, results = [], []

    header += [
        "{} MAE".format(train_args.property),
        "{} RMSE".format(train_args.property),
    ]
    derivative = model.output_modules[0].derivative
    if derivative is not None:
        header += ["{} MAE".format(derivative), "{} RMSE".format(derivative)]
    results += evaluate_dataset(metrics, model, loader, device)
    return header, results


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
