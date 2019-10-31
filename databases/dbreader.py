#! /usr/bin/env python

import os
import sys
import ase
import argparse
from ase.db import connect
from mybigdft import Posinp, Job, InputParams, Logfile
from mlcalcdriver.interfaces import ase_atoms_to_posinp
from shutil import rmtree, copyfile
from copy import deepcopy


class DbReader:
    def __init__(self):
        self.parser = self._create_parser()
        args = self.parser.parse_args()
        self.dbname = args.dbname
        self.input = InputParams()
        for filename in [f for f in os.listdir() if f.endswith(".yaml")]:
            try:
                self.input = InputParams.from_file(filename)
                break
            except:
                continue
        self.pseudos = args.no_pseudos
        self.nmpi = args.nmpi

    def _create_parser(self):
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("dbname", help="Name of the database to read.")
        parser.add_argument(
            "--no_pseudos",
            help="Use pseudos for calculations.",
            default=True,
            action="store_false",
        )
        parser.add_argument(
            "--nmpi", help="Number of mpi processes", type=int, default=6
        )
        return parser

    def read(self):
        # Create directories
        os.makedirs("run_dir/", exist_ok=True)
        os.makedirs("saved_results/", exist_ok=True)
        jobname = self.dbname.split(".")[0]

        with connect(self.dbname) as db:
            os.chdir("run_dir/")
            for i in range(1, db.count() + 1):
                at = db.get_atoms(id=i)
                pos = ase_atoms_to_posinp(at)
                try:
                    os.makedirs("{}_{:06}".format(jobname, i))
                    os.chdir("{}_{:06}".format(jobname, i))
                    job = Job(
                        name=jobname,
                        posinp=pos,
                        inputparams=self.input,
                        pseudos=self.pseudos,
                    )
                    job.run(nmpi=self.nmpi)
                    copyfile(
                        "forces_{}.xyz".format(jobname),
                        "../../saved_results/{:06}.xyz".format(i),
                    )
                    os.chdir("../")
                except OSError:
                    os.chdir("{}_{:06}".format(jobname, i))
                    try:
                        log = Logfile.from_file("log-" + jobname + ".yaml")
                        copyfile(
                            "forces_{}.xyz".format(jobname),
                            "../../saved_results/{:06}.xyz".format(i),
                        )
                        print("Calculation {:06} was complete.\n".format(i))
                    except:
                        job = Job(
                            name=jobname,
                            posinp=pos,
                            inputparams=self.input,
                            pseudos=self.pseudos,
                        )
                        job.run(nmpi=self.nmpi, restart_if_incomplete=True)
                        copyfile(
                            "forces_{}.xyz".format(jobname),
                            "../../saved_results/{:06}.xyz".format(i),
                        )
                    os.chdir("../")
            os.chdir("../")

    @property
    def dbname(self):
        return self._dbname

    @dbname.setter
    def dbname(self, dbname):
        dbname = str(dbname)
        if dbname.endswith(".db"):
            self._dbname = dbname
        else:
            self._dbname = dbname + ".db"

    @property
    def parser(self):
        return self._parser

    @parser.setter
    def parser(self, parser):
        self._parser = parser


if __name__ == "__main__":
    dbr = DbReader()
    dbr.read()
