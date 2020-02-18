#!/usr/bin/env python

import os
import csv
import torch
import argparse
import mlcalcdriver
from mlcalcdriver import Posinp, Job


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dbpath", help="Path to the database to evaluate.")
    parser.add_argument("modelpath", help="Path to the trained model.")
    parser.add_argument(
        "--overwrite", help="Remove previous evaluation file", action="store_true"
    )
    parser.add_argument("--cuda", help="Use GPU for evaluation", action="store_true")
    return parser

def main(args):
    pass

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
