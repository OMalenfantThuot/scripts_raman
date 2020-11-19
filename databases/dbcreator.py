#! /usr/bin/env python

import os
import ase
import sys
import argparse
import numpy as np
from ase.db import connect
from ase.io import read
from mlcalcdriver import Posinp
from mlcalcdriver.interfaces import posinp_to_ase_atoms
from abipy.abio.outputs import AbinitOutputFile
from utils.global_variables import DEFAULT_METADATA


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("dbname", help="Name of the database to create.")
    parser.add_argument(
        "run_mode",
        choices=["abinit", "bigdft"],
        help="Run mode used when generating the data.",
    )
    return parser


def main(args):
    with connect(args.dbname) as db:
        db.metadata = DEFAULT_METADATA
        if args.run_mode == "bigdft":
            files = [f for f in os.listdir() if f.endswith(".xyz")]
            for f in files:
                atoms = posinp_to_ase_atoms(Posinp.from_file(f))
                with open(f, "r") as posinp_file:
                    energy = float(posinp_file.readline().split()[2]) * 27.21138602
                    forces = None
                    for line in posinp_file:
                        if "forces" not in line:
                            continue
                        else:
                            forces = []
                            for _ in range(len(atoms)):
                                forces.append(
                                    [
                                        float(force)
                                        for force in posinp_file.readline().split()[-3:]
                                    ]
                                )
                            forces = np.array(forces) * 27.21138602 / 0.529177249
                            break
                db.write(atoms, data={"energy": energy, "forces": forces})

        elif args.run_mode == "abinit":
            files = [f for f in os.listdir() if f.endswith(".out")]
            for f in files:
                atoms = read(f, format="abinit-out")
                about = AbinitOutputFile(f)
                energy = float(about.final_vars_global["etotal"]) * 27.21138602
                forces = (
                    np.array(
                        [float(g) for g in about.final_vars_global["fcart"].split()]
                    ).reshape(-1, 3)
                    * 27.21138602
                    / 0.529177249
                )
                db.write(atoms, data={"energy": energy, "forces": forces})


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
