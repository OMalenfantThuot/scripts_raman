#! /usr/bin/env python

import argparse
from ase.db import connect


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
            idx1, idx2 = range(1, self.split + 1), range(self.split + 1, db.count() + 1)
            for idx in idx1:
                row = db.get(id=idx)
                db1.write(row)
            for idx in idx2:
                row = db.get(id=idx)
                db2.write(row)


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
