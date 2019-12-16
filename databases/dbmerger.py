#! /usr/bin/env python

import argparse
import sqlite3
from ase.db import connect


class DbMerger:
    def __init__(self, dbname, old_names):
        self.dbname = str(dbname)
        self.old_names = list(old_names)

    def mergedata(self):
        meta = {}
        with connect(self.dbname) as db:
            for name in self.old_names:
                with connect(name) as olddb:
                    meta.update(**olddb.metadata)
                    for row in olddb.select():
                        try:
                            db.write(row)
                        except sqlite3.IntegrityError:
                            pass
            db.metadata = meta


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("dbname", help="Path to the database to create")
    parser.add_argument("old_dbs", help="Paths to the databases to merge", nargs="*")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    dbs = DbMerger(dbname=args.dbname, old_names=args.old_dbs)
    dbs.mergedata()
