#! /usr/bin/env python

import argparse
import numpy as np
from ase.db import connect
from copy import deepcopy


class DbSplitter:
    def __init__(self, dbname, split):
        self.dbname = str(dbname)
        self.split = int(split)

    def splitdata(self):
        out1, out2 = (
            self.dbname.strip(".db") + "_1.db",
            self.dbname.strip(".db") + "_2.db",
        )
        with connect(self.dbname) as db, connect(out1) as db1, connect(out2) as db2:
            meta = deepcopy(db.metadata)
            db1.metadata = meta
            db2.metadata = meta
            idx = np.arange(1, db.count() + 1)
            np.random.shuffle(idx)
            idx1, idx2 = idx[: self.split], idx[self.split :]
            for idx in idx1:
                row = db.get(id=idx.item())
                db1.write(row)
            for idx in idx2:
                row = db.get(id=idx.item())
                db2.write(row)


class H5Splitter:
    def __init__(self, dbname, split):
        self.dbname = str(dbname)
        self.split = int(split)

    def splitdata(self):
        import h5py

        out1 = self.dbname.strip(".h5") + "_1.h5"
        out2 = self.dbname.strip(".h5") + "_2.h5"

        with h5py.File(self.dbname, "r") as old, h5py.File(
            out1, "w"
        ) as new1, h5py.File(out2, "w") as new2:
            for structname, structval in old.items():
                group1 = new1.create_group(structname)
                group2 = new2.create_group(structname)

                for i, group in enumerate([group1, group2]):
                    for prop in ["cell", "atomic_numbers"]:
                        group.create_dataset(prop, data=structval[prop])

                    for prop in ["coordinates", "dielectric", "polarization", "target"]:
                        if i == 0:
                            group.create_dataset(
                                prop, data=structval[prop][: self.split]
                            )
                        elif i == 1:
                            group.create_dataset(
                                prop, data=structval[prop][self.split :]
                            )


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["db", "h5"], help="Dataset mode")
    parser.add_argument("dbname", help="Path to the database to split")
    parser.add_argument("split", help="Size of the database to extract", type=int)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    if args.mode == "db":
        dbs = DbSplitter(dbname=args.dbname, split=args.split)
        dbs.splitdata()
    elif args.mode == "h5":
        h5s = H5Splitter(dbname=args.dbname, split=args.split)
        h5s.splitdata()
    else:
        raise ValueError("Mode is not specified.")
