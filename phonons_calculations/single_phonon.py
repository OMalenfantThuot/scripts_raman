#! /usr/bin/env python

import sys
import argparse

sys.path.append("/lustre03/project/6004866/olimt/raman/scripts_raman/")

from mybigdft import Job, Posinp, InputParams
from mybigdft.workflows import Geopt, Phonons, InfraredSpectrum
import utils
from numpy import save


def single_phonon_calculation(
    nmpi=1, nomp=1, preparation=True, savefile=True, pseudos=False
):
    if preparation:
        base_inp, ref_pos, jobname = utils.prepare_calculations()

    base_job = Job(
        posinp=ref_pos,
        inputparams=base_inp,
        name=jobname,
        run_dir="geopt/",
        pseudos=pseudos,
    )

    geopt = Geopt(base_job, forcemax=2e-5, ncount_cluster_x=50)
    geopt.run(nmpi=nmpi, nomp=nomp, restart_if_incomplete=True)

    relaxed_pos = geopt.final_posinp
    if "output" in base_inp:
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


def create_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--nmpi", help="Number of mpi processes", type=int, default=6)
    parser.add_argument("--save", help="Create a savefile", action="store_true", default=False)
    parser.add_argument("--no_pseudos", default=True, action="store_false")
    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    single_phonon_calculation(nmpi=args.nmpi, savefile=args.save, pseudos=args.no_pseudos)
