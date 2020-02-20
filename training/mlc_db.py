#!/usr/bin/env python

import os
import csv
import torch
import argparse
import numpy as np
import mlcalcdriver
from ase.db import connect
from mlcalcdriver import Posinp, Job


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dbpath", help="Path to the database to evaluate.")
    parser.add_argument("modelpath", help="Path to the trained model.")
    parser.add_argument(
        "--overwrite", help="Remove previous evaluation file", action="store_true"
    )
    parser.add_argument("--cuda", help="Use GPU for evaluation", action="store_true")
    parser.add_argument("properties", help="Properties to evaluate.", nargs="*")
    return parser


def main(args):

    eval_name = args.modelpath.split("/")[-1]
    eval_file = "evaluation_" + eval_name + ".txt"
    if os.path.exists(eval_file):
        if args.overwrite:
            os.remove(eval_file)
        else:
            raise Exception(
                "The evaluation file already exists. Delete it or add the overwrite flag."
            )

    calculator = mlcalcdriver.calculators.SchnetPackCalculator(args.modelpath)

    with connect(args.dbpath) as db:
        answers, results = (
            [[] for _ in range(len(args.properties))],
            [[] for _ in range(len(args.properties))],
        )
        for row in db.select():
            posinp = mlcalcdriver.interfaces.ase_atoms_to_posinp(row.toatoms())
            job = Job(posinp=posinp, calculator=calculator)
            for i, prop in enumerate(args.properties):
                job.run(prop)
                results[i].append(job.results[prop])
                answers[i].append(row.data[prop])

    error = []
    header = ["MAE_" + prop for prop in args.properties]
    for l1, l2 in zip(answers, results):
        an, re = np.array(l1).flatten(), np.array(l2).flatten()
        error.append(np.abs(an - re).mean())

    with open(eval_file, "w") as file:
        wr = csv.writer(file)
        wr.writerow(header)
        wr.writerow(error)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
