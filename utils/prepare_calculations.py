from mybigdft import Job, Posinp, InputParams
import os


def prepare_calculations(overwrite=False):
    r"""
    Determines which files are present and which need to be generated.

    Parameters:
    -------------
    overwrite : bool
        (default : False) if True, the present files are ignored

    Returns:
        input_is_present : bool
        True if there is an input file ("input.yaml")

        posinp_is_present : bool
        True if there is a position file

        jobname : string
        Name of the calculation if there is a pos file,
        None if no position file.
    """
    files = [f for f in os.listdir() if f.endswith((".xyz", ".yaml"))]
    input_is_present = verify_input(files)
    posinp_is_present, jobname = verify_posinp(files)
    return input_is_present, posinp_is_present, jobname


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
