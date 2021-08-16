import numpy as np
from abipy.dfpt.ddb import DdbFile
from ase import Atoms
from ase.build import make_supercell
import warnings
import shutil
from copy import deepcopy


class DDB(DdbFile):
    r"""
    Custom object to add some functionalities to the abipy
    version of the DdbFile.
    """
    def __init__(self, filepath):
        super(DDB, self).__init__(filepath)
        self.values = {}
        for i in range(len(self.qpoints)):
            self.values[i] = None

    def get_prim_mode(self, qpoint, branch):

        q_idx = np.where(np.all(self.qpoints.to_array() == qpoint, axis=1))[0]
        if q_idx.size == 0:
            raise ValueError("Asked for qpoint is not in the database.")
        else:
            q_idx = int(q_idx)

        if self.values[q_idx] is None:
            shutil.rmtree("./tmp", ignore_errors=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                bands = self.anaget_phmodes_at_qpoint(
                    qpoint,
                    asr=1,
                    chneut=1,
                    dipdip=1,
                    workdir="./tmp",
                    verbose=0,
                    anaddb_kwargs={"eivec": 1},
                )
            self.values[q_idx] = bands.dyn_mat_eigenvect[0][
                np.argsort(bands.phfreqs)[0]
            ]

        prim_displacement = self.values[q_idx][int(branch)]
        return prim_displacement


class GrapheneDDB(DDB):
    r"""
    Specific subclass to work with graphene supercells.
    """
    def __init__(self, filepath, supercell_size):
        prim = Atoms(
            symbols="C2",
            cell=[
                [2.4658461507676215, 0.0, 0.0],
                [1.2329230753838107, 2.135485408377888, 0.0],
                [0.0, 0.0, 10.0],
            ],
            positions=[[0.0, 0.0, 5.0], [1.23292308, 0.71182847, 5.0]],
            pbc=True,
        )
        atoms = make_supercell(
            prim, [[supercell_size[0], 0, 0], [0, supercell_size[1], 0], [0, 0, 1]]
        )
        self.supercell_size = supercell_size
        self.atoms = atoms
        super(GrapheneDDB, self).__init__(filepath)


    def build_supercell_modes(self, qpoint, branch, amplitudes=None):
        r"""
        Returns the Atoms with the perburbations corresponding to a specific qpoint
        and branch. Will return one perturbation for each amplitude given.
        """
        qpoint = np.array(qpoint)
        if amplitudes is None:
            amplitudes = [0.05]

        prim_displacement = self.get_prim_mode(qpoint, branch)

        lattice = self.structure.lattice.matrix
        a1, a2 = lattice[0], lattice[1]

        pieces = []
        for m in range(self.supercell_size[0]):
            for n in range(self.supercell_size[1]):
                rvec = a1 * m + a2 * n
                piece = prim_displacement * np.exp(1j * np.dot(qpoint, rvec))
                pieces.append(piece.reshape(-1, 3))
        supercell_displacement = np.concatenate(pieces).real

        modes = []
        for amp in amplitudes:
            new_positions = (
                self.atoms.positions
                + amp * supercell_displacement / np.max(supercell_displacement)
            )
            new_atoms = deepcopy(self.atoms)
            new_atoms.positions = new_positions
            modes.append(new_atoms)
        return modes
