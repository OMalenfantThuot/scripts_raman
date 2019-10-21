#! /usr/bin/env python

from mybigdft import Posinp, InputParams, Job, Logfile
from mybigdft.workflows import Geopt
from shutil import rmtree, copyfile
from copy import deepcopy
import numpy as np
import argparse
import os


class DataGenerator:
    r"""
    Drives DFT calculations by moving atoms around the equilibrium
    positions.
    """

    def __init__(self):
        self.parser = self._create_parser()
        args = self.parser.parse_args()
        self.name = args.name
        self.n_structs = args.n_structs
        self.initpos = Posinp.from_file(args.name)
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
        parser.add_argument("name", help="Name of the initial position file")
        parser.add_argument(
            "n_structs",
            type=int,
            help="Number of different structures to generate and calculate.",
        )
        parser.add_argument(
            "--no_pseudos",
            help="Use pseudos for calculations",
            default=True,
            action="store_false",
        )
        parser.add_argument(
            "--nmpi", help="Number of mpi processes", type=int, default=6
        )
        return parser

    def run(self):
        r"""
        """
        # Create directories
        os.makedirs("run_dir/", exist_ok=True)
        os.makedirs("saved_results/", exist_ok=True)
        jobname = self.name.split(".")[0]

        os.chdir("run_dir/")
        os.makedirs("geopt/", exist_ok=True)
        os.chdir("geopt/")
        job = Job(
            name=jobname,
            posinp=self.initpos,
            inputparams=self.input,
            pseudos=self.pseudos,
        )
        geopt = Geopt(job, forcemax=1e-04)
        geopt.run(nmpi=self.nmpi)
        copyfile("final_{}.xyz".format(jobname), "../../saved_results/000000.xyz")
        self.initpos = geopt.final_posinp
        os.chdir("../")
        for i in range(1, self.n_structs):
            try:
                os.makedirs("{}_{:06}".format(jobname, i))
                os.chdir("{}_{:06}".format(jobname, i))
                pos = self._generate_random_structure()
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
                    print("Calculation {:06} was complete.\n".format(i))
                except:
                    pos = self._generate_random_structure()
                    job = Job(
                        name=jobname,
                        posinp=pos,
                        inputparams=self.input,
                        pseudos=self.pseudos,
                    )
                    job.run(self.nmpi, restart_if_incomplete=True)
                    copyfile(
                        "forces_{}.xyz".format(jobname),
                        "../../saved_results/{:06}.xyz".format(i),
                    )
                os.chdir("../")
        os.chdir("../")

    def _generate_random_structure(self):
        pos = deepcopy(self.initpos)
        atoms_idx = np.arange(len(pos))
        n_translations = np.random.randint(len(pos)) + 1
        trans_idx = np.random.choice(atoms_idx, n_translations)
        for j in trans_idx:
            pos = pos.translate_atom(j, 0.2 * np.random.random(3) - 0.1)
        return pos

    @property
    def parser(self):
        return self._parser

    @parser.setter
    def parser(self, parser):
        self._parser = parser

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        name = str(name)
        if not os.path.exists(name):
            raise NameError("No positions file found for the given name.")
        self._name = name

    @property
    def n_structs(self):
        return self._n_structs

    @n_structs.setter
    def n_structs(self, n_structs):
        self._n_structs = int(n_structs)

    @property
    def initpos(self):
        return self._initpos

    @initpos.setter
    def initpos(self, initpos):
        self._initpos = initpos

    @property
    def pseudos(self):
        return self._pseudos

    @pseudos.setter
    def pseudos(self, pseudos):
        self._pseudos = pseudos

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, input):
        self._input = input


if __name__ == "__main__":
    gen = DataGenerator()
    gen.run()
