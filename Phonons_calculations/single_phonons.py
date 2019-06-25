import sys
sys.path.append('/lustre03/project/6004866/olimt/raman/scripts_raman/')

from mybigdft import Job, Posinp, InputParams
from mybigdft.workflows import (Geopt, Phonons, InfraredSpectrum)

import utils

input_file_is_present, posinp_file_is_present, jobname = utils.prepare_calculations()

if input_file_is_present:
    base_inp = InputParams.from_file('input.yaml')

if posinp_file_is_present:
    ref_pos = Posinp.from_file(jobname + '.xyz')

base_job = Job(posinp=ref_pos, inputparams=base_inp, name=jobname, run_dir="geopt/")

geopt = Geopt(base_job)
geopt.run(nmpi=2, nomp=1)
