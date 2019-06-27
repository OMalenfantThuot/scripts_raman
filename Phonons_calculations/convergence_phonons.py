#! /usr/bin/env python

import sys

sys.path.append("/lustre03/project/6004866/olimt/raman/scripts_raman/")
from single_phonon import single_phonon_calculation
import os
from shutil import copyfile
import utils
from mybigdft import InputParams, Posinp


def convergence_phonons(rmult_list, hgrid_list, nmpi=1, nomp=1, savefile=True):
    input_file_is_present, posinp_file_is_present, jobname = (
        utils.prepare_calculations()
    )

    # Cette partie devrait être dans prepare_calculations
    if input_file_is_present:
        base_inp = InputParams.from_file("input.yaml")
    else:
        base_inp = InputParams()

    if posinp_file_is_present:
        ref_pos = Posinp.from_file(jobname + ".xyz")
    elif base_inp.posinp is not None:
        ref_pos = base_inp.posinp
        jobname = "jobname"
        ref_pos.write(jobname + ".xyz")
    else:
        raise ValueError("No atomic positions are available.")

    for rmult in rmult_list:
        for hgrid in hgrid_list:
            base_dir = "rm_{:d}_{:d}/hg_{:4.2f}/".format(rmult[0], rmult[1], hgrid)
            try:
                os.makedirs(base_dir)
            except FileExistsError:
                pass
            copyfile(jobname + ".xyz", os.path.join(base_dir, jobname + ".xyz"))

            os.chdir(base_dir)
            base_inp["dft"]["rmult"] = rmult
            base_inp["dft"]["hgrids"] = hgrid
            base_inp["output"] = {"orbitals": "binary"}
            base_inp.write("input.yaml")

            single_phonon_calculation(nmpi=nmpi, nomp=nomp)
            os.chdir("../../")


if __name__ == "__main__":
    rmults = [[5, 8], [6, 9], [7, 10], [8, 11]]
    hgrids = [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]

    convergence_phonons(rmult_list=rmults, hgrid_list=hgrids, nmpi=12)
