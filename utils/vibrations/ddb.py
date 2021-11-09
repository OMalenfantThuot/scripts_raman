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
        self.modes = {}
        self.energies = {}
        for i in range(len(self.qpoints)):
            self.modes[i] = None
            self.energies[i] = None

    def get_prim_mode(self, qpoint, branch, energies=False):
        r"""
        Returns the corresponding vibration mode in the primitive unit cell.
        """

        q_idx = np.where(np.all(np.isclose(self.qpoints.to_array(), qpoint), axis=1))[0]
        if q_idx.size == 0:
            raise ValueError("Asked for qpoint is not in the database.")
        else:
            q_idx = int(q_idx)

        if self.modes[q_idx] is None:
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
            self.modes[q_idx] = bands.dyn_mat_eigenvect[0][np.argsort(bands.phfreqs)[0]]
            self.energies[q_idx] = np.sort(bands.phfreqs[0])

        prim_displacement = self.modes[q_idx][int(branch)]
        if not energies:
            return prim_displacement
        else:
            return prim_displacement, self.energies[q_idx][int(branch)]


class GrapheneDDB(DDB):
    r"""
    Specific subclass to work with graphene supercells.
    """

    def __init__(self, filepath):
        super(GrapheneDDB, self).__init__(filepath)

        size = np.sqrt(len(self.qpoints))
        assert (
            size % 1 == 0
        ), "The qpoints grid is not compatible with a homogeneous supercell."
        self.size = int(size)

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
        atoms = make_supercell(prim, [[self.size, 0, 0], [0, self.size, 0], [0, 0, 1]])
        self.atoms = atoms
        mesh = np.meshgrid(range(self.size), range(self.size))
        xlat, ylat, _ = self.structure.lattice_vectors()
        self.real_vectors = (
            mesh[1].reshape(-1, 1) * xlat + mesh[0].reshape(-1, 1) * ylat
        )

    def build_supercell_modes(
        self, qpoint, branch, amplitudes=None, return_positions=True, energies=False
    ):
        r"""
        Returns the Atoms with the perburbations corresponding to a specific qpoint
        and branch. Will return one perturbation for each amplitude given.

        Optional keywords
        -----------------
        return_positions:
            If True, returns an ase.Atoms instance of the positions. If False, only returns
            the actual perturbation in a numpy array. Default is True.

        energies:
            If True, returns the associated vibrational energy as well. Default is False.
        """
        qpoint = np.array(qpoint)
        if amplitudes is None:
            amplitudes = [0.05]
        elif not isinstance(amplitudes, (list, tuple)):
            amplitudes = [amplitudes]

        if not energies:
            prim_displacement = self.get_prim_mode(qpoint, branch)
        else:
            prim_displacement, freq = self.get_prim_mode(qpoint, branch, energies=True)

        qpoint = np.dot(qpoint, self.structure.reciprocal_lattice.matrix)

        phases = np.exp(1j * np.dot(self.real_vectors, qpoint))
        supercell_displacement = (
            prim_displacement
            * np.broadcast_to(phases.reshape(phases.shape[0], 1), (phases.shape[0], 6))
        ).reshape(-1, 3)

        modes = []
        for amp in amplitudes:
            modes.append(amp * supercell_displacement / np.max(supercell_displacement))
        if not return_positions:
            if not energies:
                return np.array(modes)[0]
            else:
                return np.array(modes)[0], freq

        else:
            outatoms = []
            for mode in modes:
                new_positions = self.atoms.positions + mode
                new_atoms = deepcopy(self.atoms)
                new_atoms.positions = new_positions
                outatoms.append(new_atoms)

            if not energies:
                return outatoms
            else:
                return outatoms, freq
