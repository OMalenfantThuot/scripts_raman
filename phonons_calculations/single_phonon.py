#! /usr/bin/env python

import sys

sys.path.append("/lustre03/project/6004866/olimt/raman/scripts_raman/")

from mybigdft import Job, Posinp, InputParams
from mybigdft.workflows import Geopt, Phonons, InfraredSpectrum
import utils
from numpy import save


def single_phonon_calculation(
    nmpi=1, nomp=1, preparation=True, savefile=True, pseudos=False
):
    if preparation:
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
        else:
            raise ValueError("No atomic positions are available.")

    base_job = Job(
        posinp=ref_pos,
        inputparams=base_inp,
        name=jobname,
        run_dir="geopt/",
        pseudos=pseudos,
    )

    geopt = Geopt(base_job)
    geopt.run(nmpi=nmpi, nomp=nomp, restart_if_incomplete=True)

    relaxed_pos = geopt.final_posinp
    if "output" in base_inp:
        # No need to write wavefunctions on disk anymore
        del base_inp["output"]
    ground_state = Job(
        name=jobname,
        posinp=relaxed_pos,
        inputparams=base_inp,
        run_dir="phonons/",
        ref_data_dir=geopt.queue[0].data_dir,
        pseudos=pseudos,
    )
    phonons = Phonons(ground_state)
    phonons.run(nmpi=nmpi, nomp=nomp, restart_if_incomplete=True)

    print("Phonons energies:")
    print(phonons.energies)

    if savefile:
        save("phonons/ph_energies.npy", phonons.energies)


if __name__ == "__main__":
    single_phonon_calculation(nmpi=2)
