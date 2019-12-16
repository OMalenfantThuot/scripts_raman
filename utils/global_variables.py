import numpy as np

__all__ = ["DB_PATH", "DEFAULT_METADATA"]

DB_PATH = "/home/olimt/projects/rrg-cotemich-ac/olimt/datasets/"

atref = np.zeros((100, 1))
atref[1, 0] = -13.6
atref[2, 0] = -75.0
atref[6, 0] = -180.0
atref[7, 0] = -278.0
atref[8, 0] = -432.0
DEFAULT_METADATA = {"atomrefs": atref.tolist(), "atref_labels": ["energy"]}
