#! /usr/bin/env python

import argparse
import ase
from ase.db import connect
import ase.io.extxyz
import h5py
import numpy as np


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dbname", help="Path to the radnet database to convert")
    parser.add_argument(
        "type", choices=["tensoap", "schnet"], help="Type of the output database."
    )
    parser.add_argument(
        "outname", help="Name of the output database (.xyz for tensoap, .db for schnet)"
    )
    return parser


def main(args):
    dbname = args.dbname + ".h5" if not args.dbname.endswith(".h5") else args.dbname

    with h5py.File(dbname, "r") as in_db:
        data = []

        for struct_name, struct_vals in in_db.items():
            for i, pos in enumerate(struct_vals["coordinates"][:]):
                data_point = {
                    "atomic_numbers": struct_vals["atomic_numbers"][:],
                    "cell": struct_vals["cell"][:],
                    "coordinates": pos,
                    "dielectric": struct_vals["dielectric"][i],
                    "polarization": struct_vals["polarization"][i],
                }
                data.append(data_point)

    atoms = []
    for point in data:
        atom = ase.Atoms(
            numbers=point["atomic_numbers"],
            cell=point["cell"],
            positions=point["coordinates"],
            pbc=True,
        )
        atom.info = {
            "dielectric": point["dielectric"].reshape(9),
            "polarization": point["polarization"],
        }
        atoms.append(atom)

    if args.type == "tensoap":
        outname = (
            args.outname + ".xyz" if not args.outname.endswith(".xyz") else args.outname
        )
        ase.io.extxyz.write_extxyz(outname, atoms)

    elif args.type == "schnet":
        outname = (
            args.outname + ".db" if not args.outname.endswith(".db") else args.outname
        )
        with connect(outname) as db:
            for atom in atoms:
                db.write(
                    atom,
                    data={
                        "dielectric": atom.info["dielectric"],
                        "polarization": atom.info["polarization"],
                    },
                )


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
