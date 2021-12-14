#! /usr/bin/env python

from ase.io import read
from utils.global_variables import DEFAULT_METADATA
from ase.db import connect
import argparse


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("dbname", help="Name of the database to create.")
    parser.add_argument("filename", help="Name of the .xml file to read.")
    parser.add_argument(
        "--keepstep",
        default=5,
        type=int,
        help="Keep one configuration every X steps. Default is 5.",
    )
    return parser


def main(args):
    dbname = args.dbname if args.dbname.endswith(".db") else args.dbname + ".db"
    with open(args.filename) as infile, connect(dbname) as db:
        db.metadata = DEFAULT_METADATA
        atomslist = read(args.filename, index=":", format="vasp-xml")

        for i, atoms in enumerate(atomslist):
            if (i + 1) % args.keepstep == 0:
                energy = atoms._calc.results["energy"]
                forces = atoms._calc.results["forces"]
                atoms.calc = None
                db.write(atoms, data={"energy": energy, "forces": forces})
            else:
                continue


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
