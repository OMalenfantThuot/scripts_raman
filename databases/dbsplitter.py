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
            idx = np.arange(1, db.count() + 1)
            np.random.shuffle(idx)
            idx1, idx2 = idx[: self.split], idx[self.split :]
            for idx in idx1:
                row = db.get(id=idx.item())
                db1.write(row)
            db1.metadata = meta
            for idx in idx2:
                row = db.get(id=idx.item())
                db2.write(row)
            db2.metadata = meta


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dbname", help="Path to the database to split")
    parser.add_argument("split", help="Size of the database to extract", type=int)
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    dbs = DbSplitter(dbname=args.dbname, split=args.split)
    dbs.splitdata()
