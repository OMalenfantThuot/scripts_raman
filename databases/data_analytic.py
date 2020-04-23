#! /usr/bin/env python

import sys
import argparse
import numpy as np
from ase.db import connect
from mlcalcdriver import Posinp
from mlcalcdriver.interfaces import posinp_to_ase_atoms


def main(args):
    function = get_function(args.function)
    distances = np.linspace(args.range[0], args.range[1], args.ndata)
    with connect(args.dbname) as db:
        for d in distances:
            pos_dict = {
                "units": "angstroem",
                "boundary_conditions": "free",
                "positions": [{args.element: [0, 0, 0]}, {args.element: [d, 0, 0]},],
            }
            posinp = Posinp.from_dict(pos_dict)
            atoms = posinp_to_ase_atoms(posinp)
            energy = function.value(d)
            forces = np.array(
                [
                    [-1.0 * function.first_derivative(-d), 0, 0],
                    [-1.0 * function.first_derivative(d), 0, 0],
                ]
            )
            db.write(atoms, data={"energy": energy, "forces": forces})


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dbname", help="Name of the database.")
    parser.add_argument("ndata", type=int, help="Number of data points to generate.")
    parser.add_argument(
        "--element",
        default="C",
        help="Element to use for the pair of atoms. Default is C.",
    )
    parser.add_argument(
        "--function",
        default="cos",
        help="Analytic function defining the potential. Default is a sinus.",
    )
    parser.add_argument(
        "--range",
        default=[0.1, 1.0],
        type=float,
        help="Range for the interatomic distance for which data will be generated.",
        nargs=2,
    )
    return parser


def get_function(name):
    if name == "sin":
        from utils.functions.analytic_functions import Sin

        return Sin()
    elif name == "cos":
        from utils.functions.analytic_functions import Cos

        return Cos()
    elif name == "lj" or name == "lennardjones":
        from utils.functions.analytic_functions import LennardJones

        return LennardJones()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
