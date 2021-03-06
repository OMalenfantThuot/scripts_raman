#! /usr/bin/env python

import sys

sys.path.append("/lustre03/project/6004866/olimt/raman/scripts_raman/")
from phonons_calculations import single_phonon_calculation
import os
from shutil import copyfile
import utils
from mybigdft import InputParams, Posinp
import numpy as np


def convergence_phonons(
    rmult_list, hgrid_list, nmpi=1, nomp=1, savefile=True, pseudos=False
):
    base_inp, ref_pos, jobname = utils.prepare_calculations()

    for rmult in rmult_list:
        for hgrid in hgrid_list:
            base_dir = "rm_{:d}_{:d}/hg_{:4.2f}/".format(rmult[0], rmult[1], hgrid)
            try:
                os.makedirs(base_dir)
            except FileExistsError:
                pass
            copyfile(jobname + ".xyz", os.path.join(base_dir, jobname + ".xyz"))

            os.chdir(base_dir)
            try:
                base_inp["dft"]["rmult"] = rmult
                base_inp["dft"]["hgrids"] = hgrid
            except KeyError:
                base_inp["dft"] = {"rmult": rmult, "hgrids": hgrid}
            base_inp["output"] = {"orbitals": "binary"}
            base_inp.write("input.yaml")

            single_phonon_calculation(
                nmpi=nmpi, nomp=nomp, savefile=savefile, pseudos=pseudos
            )
            os.chdir("../../")

    # Temporary solution, should probably be done in sqlite
    if savefile:
        results = np.zeros((len(rmult_list), len(hgrid_list), 3 * len(ref_pos)))
        for i, rmult in enumerate(rmult_list):
            for j, hgrid in enumerate(hgrid_list):
                datadir = "rm_{:d}_{:d}/hg_{:4.2f}/phonons/".format(
                    rmult[0], rmult[1], hgrid
                )
                results[i][j] = np.load(datadir + "ph_energies.npy")
        np.save(jobname + "_ph_energies.npy", results)


if __name__ == "__main__":
    rmults = [[5, 8], [6, 9], [7, 10], [8, 11]]
    hgrids = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]

    convergence_phonons(rmult_list=rmults, hgrid_list=hgrids, nmpi=12)
