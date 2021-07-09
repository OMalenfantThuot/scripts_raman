#! /usr/bin/env python

from abipy.abio.outputs import AbinitOutputFile
from ase.io import read
from ase.calculators.abinit import Abinit
from shutil import rmtree, copyfile
from copy import deepcopy
import yaml
import numpy as np
import argparse
import pickle
import os


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "n_data_per_mode",
        type=int,
        help="Number of times each mode is used to generate a structure.",
    )
    parser.add_argument(
        "positions", help="Name of the initial positions file(abinit format)"
    )
    parser.add_argument("input", help="Name of the yaml input file.")
    parser.add_argument(
        "modes", help="Name of the pickle file containing the normal modes."
    )
    return parser


def main(args):
    # Create directories
    os.makedirs("run_dir/", exist_ok=True)
    os.makedirs("saved_results/", exist_ok=True)

    with open(args.modes, "rb") as f:
        modes = pickle.load(f)
    modes = modes[0]

    jobname = args.positions.split(".")[0]
    initatoms = read(args.positions, format="abinit-in")
    with open(args.input, "r") as f:
        inputs = yaml.load(f, Loader=yaml.BaseLoader)
    calculator = Abinit(**inputs)

    os.chdir("run_dir/")
    for i in range(args.n_data_per_mode):
        for j, mode in enumerate(modes):
            try:
                os.makedirs("{}_{:03}_mode{:04}".format(jobname, i, j))
                os.chdir("{}_{:03}_mode{:04}".format(jobname, i, j))
                abinit_run(i, j, mode, calculator, initatoms)
            except FileExistsError:
                os.chdir("{}_{:03}_mode{:04}".format(jobname, i, j))
                try:
                    about = AbinitOutputFile("abinit.out")
                    if about.run_completed:
                        print(
                            "Calculation {:03} for mode {:04} was complete.\n".format(
                                i, j
                            )
                        )
                        os.chdir("../")
                    else:
                        restart(i, j, mode, calculator, initatoms)
                except:
                    restart(i, j, mode, calculator, initatoms)


def restart(i, j, mode, calculator, initatoms):
    for f in os.listdir():
        os.remove(f)
    abinit_run(i, j, mode, calculator, initatoms)


def abinit_run(i, j, mode, calculator, initatoms):
    at = abinit_structure(initatoms, mode)
    at.set_calculator(calculator)
    at.get_forces()
    os.rename("abinit.txt", "abinit.out")
    copyfile("abinit.out", "../../saved_results/{:03}_mode{:04}.out".format(i, j))
    for f in os.listdir():
        if f.startswith("abinito_"):
            os.remove(f)
    print("Calculation {:03} for mode {:04} completed.\n".format(i, j))
    os.chdir("../")


def abinit_structure(initatoms, mode):
    at = deepcopy(initatoms)
    at.set_positions(at.positions + np.real(mode) * 2 * np.random.rand())
    return at


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
