#! /usr/bin/env python

from mybigdft import Posinp, InputParams, Job, Logfile, Atom
from shutil import rmtree, copyfile
from copy import deepcopy
import numpy as np
import argparse
import sys
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
        
        initpos, _ = place_first_nitrogen(initpos)
        
        for i in range(1, args.n_structs + 1):
            posinp = generate_random_structure(initpos)
            run(posinp, i, args, param, pseudos)

    elif args.n_defects == 2:

        initpos, first_idx = place_first_nitrogen(initpos)

        root = np.sqrt(args.n_structs)
        if root % 1 == 0:
            n_angle, n_radius = root, root
        else:
            n_angle, n_radius = np.ceil(root), np.floor(root)
        max_radius = np.max(np.array(initpos.cell) / 2)
        radiuses = np.linspace(1.0, max_radius, n_radius)
        angles = np.linspace(0, 2 * np.pi, n_angle)

        i = 0
        #distances = np.zeros(int(n_angle * n_radius))
        for theta in angles:
            for r in radiuses:
                i += 1 
                posinp, second_idx = place_second_nitrogen(initpos, theta, r, first_idx)
        #        distances[i-1] = np.linalg.norm(posinp.positions[first_idx] - posinp.positions[second_idx])
                run(posinp, i, args, param, pseudos)
        #np.savetxt("distances.data", distances)

    elif args.n_defects == 3:
        root = np.sqrt(args.n_structs)
        if root % 1 == 0:
            n_angle, n_radius = root, root
        else:
            n_angle, n_radius = np.ceil(root), np.floor(root)

    else:
        raise NotImplementedError("No method for this number of defects.")


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
        r = radius * np.cbrt(np.random.rand())
        pos = pos.translate_atom(
            j,
            [
                r * np.sin(theta) * np.cos(phi),
                r * np.sin(theta) * np.sin(phi),
                r * np.cos(theta),
            ],
        )
    return pos

def place_first_nitrogen(posinp):
    distances_to_middle = np.linalg.norm(
        posinp.positions - np.array(posinp.cell) / 2, axis=1
    )
    idx = np.argmin(distances_to_middle)
    posinp.atoms[idx] = Atom("N", posinp.atoms[idx].position)
    return posinp, idx

def place_second_nitrogen(initpos, theta, r, first_idx):
    posinp = deepcopy(initpos)
    point = np.array(posinp.cell) / 2 + np.array([r * np.cos(theta), 0, r * np.sin(theta)])
    distances_to_point = np.linalg.norm(posinp.positions - point, axis=1)
    weigths = np.exp(-distances_to_point ** 2 / 2)
    weigths[first_idx] = 0
    weigths = weigths/np.sum(weigths)
    second_idx = int(np.random.choice(len(posinp), 1, p=weigths))
    posinp.atoms[second_idx] = Atom("N", posinp.atoms[second_idx].position)
    return posinp, second_idx


def create_parser():
    parser = argparse.ArgumentParser()
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
