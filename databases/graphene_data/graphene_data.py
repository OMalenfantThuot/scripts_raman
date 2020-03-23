#! /usr/bin/env python

from mybigdft import Posinp, InputParams, Job, Logfile
from shutil import rmtree, copyfile
from copy import deepcopy
import numpy as np
import argparse
import os

sys.path.append("/lustre03/project/6004866/olimt/raman/scripts_raman/")
from utils import generate_graphene_cell


def main(args):

    initpos = generate_graphene_cell(args.xsize, args.zsize)
    param = InputParams()
    for filename in [f for f in os.listdir() if f.endswith(".yaml")]:
        try:
            param = InputParams.from_file(filename)
            break
        except:
            continue
    pseudos = not (args.no_pseudos)

    # Create directories
    os.makedirs("run_dir/", exist_ok=True)
    os.makedirs("saved_results/", exist_ok=True)

    os.chdir("run_dir/")

    if args.n_defects == 0:
        for i in range(1, args.n_structs + 1):
            posinp = generate_random_structure(initpos)
            run(posinp, i, args, param, pseudos)
    elif args.n_defects == 1:
        pass
    elif args.n_defects == 2:
        pass
    elif args.n_defects == 3:
        pass
    elif raise NotImplementedError("No method for this number of defects.")

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
        phi = np.random.rand(0, 2 * np.pi)
        theta = np.arccos(np.random.rand(-1.0, 1.0))
        r = radius * np.cubrt(np.random.rand(0.0, 1.0))
        pos = pos.translate_atom(
            j,
            [
                r * np.sin(theta) * cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ],
        )
    return pos


def create_parser(self):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "name", type=str, help="Name to give to calculations.", default="Jobname"
    )
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
    parser.add_argument("--nmpi", help="Number of mpi processes", type=int, default=6)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
