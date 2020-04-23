#! /usr/bin/env python

from mybigdft import Posinp, InputParams, Job, Logfile
from shutil import rmtree, copyfile
from copy import deepcopy
import numpy as np
import argparse
import os


def main(args):

    param = InputParams()
    for filename in [f for f in os.listdir() if f.endswith(".yaml")]:
        try:
            param = InputParams.from_file(filename)
            break
        except:
            continue
    pseudos = not (args.no_pseudos)

    positions = [
        Posinp.from_file(args.positions_dir + file)
        for file in os.listdir(args.positions_dir)
        if file.endswith(".xyz")
    ]

    # Create directories
    os.makedirs("run_dir/", exist_ok=True)
    os.makedirs("saved_results/", exist_ok=True)

    os.chdir("run_dir/")

    for i in range(args.n_structs):
        j = int(np.random.choice(len(positions), 1))
        posinp = generate_random_structure(positions[j])
        run(posinp, i, args, param, pseudos)


def run(posinp, i, args, param, pseudos):
    try:
        os.makedirs("{}_{:06}".format(args.name, i))
        os.chdir("{}_{:06}".format(args.name, i))
        job = Job(name=args.name, posinp=posinp, inputparams=param, pseudos=pseudos)
        job.run(nmpi=args.nmpi)
        copyfile(
            "forces_{}.xyz".format(args.name),
            "../../saved_results/{:06}.xyz".format(i),
        )
        os.chdir("../")
    except OSError:
        os.chdir("{}_{:06}".format(args.name, i))
        try:
            log = Logfile.from_file("log-" + args.name + ".yaml")
            print("Calculation {:06} was complete.\n".format(i))
        except:
            job = Job(name=args.name, posinp=posinp, inputparams=param, pseudos=pseudos)
            job.run(args.nmpi, restart_if_incomplete=True)
            copyfile(
                "forces_{}.xyz".format(args.name),
                "../../saved_results/{:06}.xyz".format(i),
            )
        os.chdir("../")


def generate_random_structure(initpos):
    radius = 0.1
    pos = deepcopy(initpos)
    atoms_idx = np.arange(len(pos))
    n_translations = np.random.randint(len(pos)) + 1
    trans_idx = np.random.choice(atoms_idx, n_translations)
    for j in trans_idx:
        phi = 2 * np.pi * np.random.rand()
        theta = np.arccos(2 * np.random.rand() - 1)
        r = radius * np.random.rand()
        pos = pos.translate_atom(
            j,
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ],
        )
    return pos


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "name", type=str, help="Name to give to calculations.", default="Jobname"
    )
    parser.add_argument(
        "positions_dir", type=str, help="Folder containing the configurations to sample"
    )
    parser.add_argument(
        "n_structs",
        type=int,
        help="Number of different structures to generate and calculate.",
    )
    parser.add_argument(
        "--no_pseudos",
        help="Add to conduct the calculations without pseudopotentials.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--nmpi", help="Number of mpi processes, default is 6.", type=int, default=6
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
