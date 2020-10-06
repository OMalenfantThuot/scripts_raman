#! /usr/bin/env python

import argparse
import numpy as np
from utils.datagen import create_2d_square_data


def main(args):
    create_2d_square_data(args.ndata, args.size, args.name)


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("size", type=int, help="Supercell size in one direction.")
    parser.add_argument("ndata", type=int, help="Number of configurations to generate.")
    parser.add_argument("--name", default="data.db", help="Name of the db file.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
