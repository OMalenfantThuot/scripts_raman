r"""
Functions to work with the ase database objects
"""

import ase
import os
from glob import glob
from global_variables import DB_PATH
import mybigdft as mb

__all__ = ["list_available_db"]


def list_available_db(PATH=DB_PATH):
    r"""
    Search for available database objects in PATH

    Parameters:
    ------------
    PATH : string
        path in which to search for databases

    Returns:
        list of paths of database objects
    """
    return [db for f in os.walk(PATH) for db in glob(os.path.join(f[0], "*.db"))]
