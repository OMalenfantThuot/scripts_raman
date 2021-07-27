#!/usr/bin/env python

import os
import csv
import torch
import argparse
import numpy as np
import mlcalcdriver
from ase.db import connect
from mlcalcdriver import Posinp, Job
import pickle


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dbpath", help="Path to the database to evaluate.")
    parser.add_argument("properties", help="Properties to evaluate.", nargs="*")
    parser.add_argument(
        "--models", help="Paths to the trained models.", default=[], nargs="*"
    )
    parser.add_argument("--name", help="Name of the evaluation file.", default=None)
    parser.add_argument(
        "--overwrite", help="Remove previous evaluation file", action="store_true"
    )
    parser.add_argument("--cuda", help="Use GPU for evaluation", action="store_true")
    parser.add_argument(
        "--detailed", action="store_true", help="Get the values for each data point."
    )
    return parser


def main(args):

    eval_name = args.models[0].split("/")[-1] if args.name is None else args.name
    eval_file = "evaluation_" + eval_name + ".txt"
    if os.path.exists(eval_file):
        if args.overwrite:
            os.remove(eval_file)
        else:
            raise Exception(
                "The evaluation file already exists. Delete it or add the overwrite flag."
            )
    device = "cuda" if args.cuda else "cpu"

    calculator = mlcalcdriver.calculators.EnsembleCalculator(args.models, device=device)

    with connect(args.dbpath) as db:
        answers, results, stds = (
            [[] for _ in range(len(args.properties))],
            [[] for _ in range(len(args.properties))],
            [[] for _ in range(len(args.properties))],
        )
        for row in db.select():
            for i, prop in enumerate(args.properties):
                answers[i].append(row.data[prop])
            posinp = Posinp.from_ase(row.toatoms())

            job = Job(posinp=posinp, calculator=calculator)
            for i, prop in enumerate(args.properties):
                job.run(prop)
                results[i].append(job.results[prop].tolist())
                stds[i].append(job.results[prop + "_std"].tolist())

    header, error = [], []
    if args.detailed:
        full_err, full_std = {}, {}
    for prop in args.properties:
        header += ["MAE_" + prop, "%Error_" + prop, "STD_" + prop]
        if args.detailed:
            full_err[prop], full_std[prop] = [], []
    for p, l1, l2, l3 in zip(args.properties, answers, results, stds):
        an, re, std = (
            np.array(l1).flatten(),
            np.concatenate(l2).flatten(),
            np.concatenate(l3).flatten(),
        )
        error.append(np.abs(an - re).mean())
        error.append(np.abs((an - re) / an).mean() * 100)
        error.append(np.mean(std))
        if args.detailed:
            full_err[p].append(np.abs(an - re))
            full_std[p].append(std)

    with open(eval_file, "w") as file:
        wr = csv.writer(file)
        wr.writerow(header)
        wr.writerow(error)
    if args.detailed:
        with open(eval_name + "_err.pkl", "wb") as f:
            pickle.dump(full_err, f)
        with open(eval_name + "_std.pkl", "wb") as f:
            pickle.dump(full_std, f)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
