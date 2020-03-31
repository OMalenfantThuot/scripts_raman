#! /usr/bin/env python

from mybigdft import Posinp, Atom
from copy import deepcopy
import numpy as np
import argparse
import sys

sys.path.append("/lustre03/project/6004866/olimt/raman/scripts_raman/")
from utils import generate_graphene_cell, determine_unique_configurations


def main(args):

    initpos = generate_graphene_cell(args.xsize, args.zsize)
    configurations = []

    if args.n_defects == 1:

        initpos, _ = place_first_nitrogen(initpos)
        configurations = [initpos]

    elif args.n_defects == 2:

        initpos, first_idx = place_first_nitrogen(initpos)
        for i, at in enumerate(initpos):
            new_pos = deepcopy(initpos)
            if i != first_idx:
                new_pos.atoms[i] = Atom("N", at.position)
                configurations.append(new_pos)

    elif args.n_defects == 3:

        initpos, first_idx = place_first_nitrogen(initpos)
        for i, at1 in enumerate(initpos):
            if i != first_idx:
                new_pos = deepcopy(initpos)
                new_pos.atoms[i] = Atom("N", at1.position)
                for j, at2 in enumerate(initpos):
                    if j != first_idx and j > i:
                        new_pos2 = deepcopy(new_pos)
                        new_pos2.atoms[j] = Atom("N", at2.position)
                        configurations.append(new_pos2)

    unique_configurations = determine_unique_configurations(configurations)
    print("There are {} unique configurations.".format(len(unique_configurations)))
    if args.output:
        for i, uni in enumerate(unique_configurations):
            uni.write("{}_{:03}.xyz".format(args.name, i))


def place_first_nitrogen(posinp):
    distances_to_middle = np.linalg.norm(
        posinp.positions - np.array(posinp.cell) / 2, axis=1
    )
    idx = np.argmin(distances_to_middle)
    posinp.atoms[idx] = Atom("N", posinp.atoms[idx].position)
    return posinp, idx


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="Name for outputs.", default="name")
    parser.add_argument(
        "n_defects", type=int, help="Number of nitrogen defects to introduce."
    )
    parser.add_argument(
        "xsize",
        type=int,
        help="Number of repetition of the base cell in the x direction.",
    )
    parser.add_argument(
        "zsize",
        type=int,
        help="Number of repetition of the base cell in the z direction.",
    )
    parser.add_argument(
        "--output",
        default=False,
        action="store_true",
        help="If used, the unique positions will be written on disk.",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
