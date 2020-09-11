#! /usr/bin/env python

from mlcalcdriver.base import Posinp, Atom
from copy import deepcopy
import numpy as np
import argparse
import sys
from utils.calculations.graphene import generate_graphene_cell
from utils.calculations.configurations import determine_unique_configurations


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

    elif args.n_defects == 4:

        initpos, first_idx = place_first_nitrogen(initpos)
        for i, at1 in enumerate(initpos):
            if i != first_idx:
                new_pos = deepcopy(initpos)
                new_pos.atoms[i] = Atom("N", at1.position)
                for j, at2 in enumerate(initpos):
                    if j != first_idx and j > i:
                        new_pos2 = deepcopy(new_pos)
                        new_pos2.atoms[j] = Atom("N", at2.position)
                        for k, at3 in enumerate(initpos):
                            if k != first_idx and k > j:
                                new_pos3 = deepcopy(new_pos2)
                                new_pos3.atoms[k] = Atom("N", at3.position)
                                configurations.append(new_pos3)

    elif args.n_defects == 5:

        initpos, first_idx = place_first_nitrogen(initpos)
        for i, at1 in enumerate(initpos):
            if i != first_idx:
                new_pos = deepcopy(initpos)
                new_pos.atoms[i] = Atom("N", at1.position)
                for j, at2 in enumerate(initpos):
                    if j != first_idx and j > i:
                        new_pos2 = deepcopy(new_pos)
                        new_pos2.atoms[j] = Atom("N", at2.position)
                        for k, at3 in enumerate(initpos):
                            if k != first_idx and k > j:
                                new_pos3 = deepcopy(new_pos2)
                                new_pos3.atoms[k] = Atom("N", at3.position)
                                for m, at4 in enumerate(initpos):
                                    if m != first_idx and m > k:
                                        new_pos4 = deepcopy(new_pos3)
                                        new_pos4.atoms[m] = Atom("N", at4.position)
                                        configurations.append(new_pos4)

    unique_configurations, count_configurations = determine_unique_configurations(
        configurations
    )
    print(
        "There are {} unique configurations, and {} configurations in total.".format(
            len(unique_configurations), np.sum(count_configurations)
        )
    )
    print("Counter:", count_configurations)
    if args.output:
        for i, (uni, count) in enumerate(
            zip(unique_configurations, count_configurations)
        ):
            uni.write("{}_{:03}_(x{}).xyz".format(args.name, i, count))


def place_first_nitrogen(posinp):
    distances_to_middle = np.linalg.norm(
        posinp.positions
        - np.array((posinp.cell[0, 0], posinp.cell[1, 1], posinp.cell[2, 2])) / 2,
        axis=1,
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
