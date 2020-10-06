import numpy as np
from ase.calculators.lj import LennardJones
from ase import Atoms
from copy import deepcopy
from ase.db import connect


def create_2d_square_data(ndata, size, dbname):

    sigma = 1.0
    d = 2 ** (1 / 6) * sigma

    assert type(size) is int
    positions = np.zeros((size ** 2, 3))

    x = np.array(range(size)) * d
    x1, x2 = np.meshgrid(x, x)

    positions[:, 0] = x1.reshape(-1)
    positions[:, 1] = x2.reshape(-1)

    cell = [size * d, size * d, 0]
    init_atoms = Atoms("H{}".format(size ** 2), positions=positions, cell=cell, pbc=[1,1,0])

    calculator = LennardJones(sigma=sigma)
    init_atoms.set_calculator(calculator)

    dbname = dbname if dbname.endswith(".db") else dbname + ".db"
    with connect(dbname) as db:
        for _ in range(ndata):
            atoms = deepcopy(init_atoms)
            disp = get_random_displacement(size, d)
            atoms.set_positions(atoms.positions + disp)
            db.write(
                atoms,
                data={
                    "energy": atoms.get_total_energy(),
                    "forces": atoms.get_forces(),
                    "displacements": disp,
                },
            )


def get_random_displacement(size, d):
    disp = np.zeros((size ** 2, 3))
    angles = np.random.rand(size ** 2) * 2 * np.pi
    norm = np.random.rand(size ** 2) * 0.05 * d
    disp[:, 0], disp[:, 1] = norm * np.cos(angles), norm * np.sin(angles)
    return disp
