import numpy as np
from mlcalcdriver import Posinp


def generate_graphene_cell(xsize, zsize):
    base_cell = np.array([2.4674318, 0, 4.2737150])
    positions = []
    reduced_pos = np.array(
        [[0, 0, 0], [0, 0, 1.0 / 3], [0.5, 0, 0.5], [0.5, 0, 5.0 / 6]]
    )
    for i in range(xsize):
        for j in range(zsize):
            p = (np.array([i, 0, j]) + reduced_pos) * base_cell
            for pi in p:
                positions.append({"C": pi})

    pos_dict = {
        "units": "angstroem",
        "cell": base_cell * np.array([xsize, 0, zsize]),
        "positions": positions,
    }
    pos = Posinp.from_dict(pos_dict)
    return pos
