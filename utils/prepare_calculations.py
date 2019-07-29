from mybigdft import Posinp, InputParams
import os


def prepare_calculations(overwrite=False):
    r"""
    Determines which files are present and which need to be generated.

    Parameters:
    -------------
    overwrite : bool
        (default : False) if True, the present files are ignored

    Returns:
        base_inp : InputParams
        base input parameters for the calculation

        base_pos : Posinp
        base atomic positions for the calculation

        jobname : string
        Name of the calculation if there is a pos file,
        string "jobname" if no position file.
    """
    files = [f for f in os.listdir() if f.endswith((".xyz", ".yaml"))]
    posinp_is_present, jobname = verify_posinp(files)

    if verify_input(files):
        base_inp = InputParams.from_file("input.yaml")
    else:
        base_inp = InputParams()

    if posinp_is_present:
        base_pos = Posinp.from_file(jobname + ".xyz")
    elif base_inp.posinp is not None:
        base_pos = base_inp.posinp
        jobname = "jobname"
        base_pos.write(jobname + ".xyz")
    else:
        raise ValueError("No atomic positions are available.")
    return base_inp, base_pos, jobname


def verify_input(files):
    if "input.yaml" in files:
        return True
    elif any([f.endswith(".yaml") for f in files]):
        raise NameError("The input file should be named input.yaml.")
    else:
        return False


def verify_posinp(files):
    pos_file = next((f for f in files if f.endswith(".xyz")), None)
    if pos_file:
        jobname = pos_file.strip(".xyz")
        return True, jobname
    else:
        return False, None
