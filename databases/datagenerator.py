#! /usr/bin/env python

from mybigdft import Posinp, InputParams, Job, Logfile
from mybigdft.workflows import Geopt
from abipy.abio.outputs import AbinitOutputFile
from ase.io import read
from ase.calculators.abinit import Abinit
from shutil import rmtree, copyfile
from copy import deepcopy
import yaml
import numpy as np
import argparse
import os


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "n_structs",
        type=int,
        help="Number of different structures to generate and calculate.",
    )
    runmode_subparser = parser.add_subparsers(
        dest="run_mode", help="Choose the DFT code to generate data."
    )

    # BigDFT parser
    bigdft_parser = runmode_subparser.add_parser(
        "bigdft",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="BigDFT calculations.",
    )

    bigdft_parser.add_argument("posinp", help="Name of the initial positions file.")
    bigdft_parser.add_argument(
        "--no_pseudos",
        help="Use pseudos for calculations",
        default=True,
        action="store_false",
    )
    bigdft_parser.add_argument(
        "--nmpi", help="Number of mpi processes", type=int, default=6
    )

    # Abinit parser
    abinit_parser = runmode_subparser.add_parser(
        "abinit",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Abinit calculations.",
    )
    abinit_parser.add_argument(
        "positions", help="Name of the initial positions file(abinit format)"
    )
    abinit_parser.add_argument("input", help="Name of the yaml input file.")
    return parser


def main(args):
    # Create directories
    os.makedirs("run_dir/", exist_ok=True)
    os.makedirs("saved_results/", exist_ok=True)

    if args.run_mode == "bigdft":
        jobname = args.posinp.split(".")[0]
        initpos = Posinp.from_file(args.posinp)

        inputpar = InputParams()
        for filename in [f for f in os.listdir() if f.endswith(".yaml")]:
            try:
                inputpar = InputParams.from_file(filename)
                break
            except:
                continue

        os.chdir("run_dir/")
        for i in range(args.n_structs):
            try:
                os.makedirs("{}_{:06}".format(jobname, i))
                os.chdir("{}_{:06}".format(jobname, i))
                bigdft_run(i, initpos, args, jobname, inputpar, restart=False)
            except OSError:
                os.chdir("{}_{:06}".format(jobname, i))
                try:
                    log = Logfile.from_file("log-" + jobname + ".yaml")
                    print("Calculation {:06} was complete.\n".format(i))
                    os.chdir("../")
                except:
                    bigdft_run(i, initpos, args, jobname, inputpar, restart=True)
        os.chdir("../")

    elif args.run_mode == "abinit":
        jobname = args.positions.split(".")[0]
        initatoms = read(args.positions, format="abinit-in")
        with open(args.input, "r") as f:
            inputs = yaml.load(f, Loader=yaml.BaseLoader)
        calculator = Abinit(**inputs)

        os.chdir("run_dir/")
        for i in range(args.n_structs):
            try:
                os.makedirs("{}_{:06}".format(jobname, i))
                os.chdir("{}_{:06}".format(jobname, i))
                abinit_run(i, calculator, initatoms)
            except FileExistsError:
                os.chdir("{}_{:06}".format(jobname, i))
                try:
                    about = AbinitOutputFile("abinit.out")
                    if about.run_completed:
                        print("Calculation {:06} was complete.\n".format(i))
                        os.chdir("../")
                    else:
                        restart(i, calculator, initatoms)
                except:
                    restart(i, calculator, initatoms)
    else:
        raise ValueError("The run_mode argument should be abinit or bigdft.")

def bigdft_run(i, initpos, args, jobname, inputpar, restart=False):
    pos = bigdft_random_structure(initpos)
    job = Job(
        name=jobname,
        posinp=pos,
        inputparams=inputpar,
        pseudos=args.no_pseudos,
    )
    job.run(nmpi=args.nmpi, restart_if_incomplete=restart)
    copyfile(
        "forces_{}.xyz".format(jobname),
        "../../saved_results/{:06}.xyz".format(i),
    )
    os.chdir("../")

def bigdft_random_structure(initpos):
    pos = deepcopy(initpos)
    atoms_idx = np.arange(len(pos))
    n_translations = np.random.randint(len(pos)) + 1
    trans_idx = np.random.choice(atoms_idx, n_translations)
    for j in trans_idx:
        phi = 2 * np.pi * np.random.rand()
        theta = np.arccos(2 * np.random.rand() - 1)
        r = -0.1
        while r < 0:
            r = np.random.normal(scale=0.1)
        pos = pos.translate_atom(
            j,
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ],
        )
    return pos

def restart(i, calculator, initatoms):
    for f in os.listdir():
        os.remove(f)
    abinit_run(i, calculator, initatoms)

def abinit_run(i, calculator, initatoms):
    at = abinit_random_structure(initatoms)
    at.set_calculator(calculator)
    at.get_forces()
    os.rename("abinit.txt", "abinit.out")
    copyfile("abinit.out", "../../saved_results/{:06}.out".format(i))
    for f in os.listdir():
        if f.startswith("abinito_"):
            os.remove(f)
    print("Calculation {:06} completed.\n".format(i))
    os.chdir("../")

def abinit_random_structure(initatoms):
    at = deepcopy(initatoms)
    atoms_idx = np.arange(len(at))
    n_translations = np.random.randint(len(at)) + 1
    trans_idx = np.random.choice(atoms_idx, n_translations)
    for j in trans_idx:
        phi = 2 * np.pi * np.random.rand()
        theta = np.arccos(2 * np.random.rand() - 1)
        r = -0.1
        while r < 0:
            r = np.random.normal(scale=0.1)
        at.positions[j] = at.positions[j] + np.array(
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ]
        )
    return at


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
