#! /usr/bin/env python

import argparse
import ase
import numpy as np
import os
import random
import sys
from ase.db import connect
from ase.io import read
from ase.units import Bohr
from mlcalcdriver import Posinp
from mlcalcdriver.interfaces import posinp_to_ase_atoms
from utils.global_variables import DEFAULT_METADATA, DEFAULT_MD_METADATA


def create_parser():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("dbname", help="Name of the database to create.")
    parser.add_argument(
        "run_mode",
        choices=["abinit", "bigdft", "md", "radnet"],
        help="Run mode used when generating the data.",
    )
    parser.add_argument(
        "--h5group", default=None, help="Name of the h5 group. Only for radnet."
    )
    parser.add_argument(
        "--target",
        choices=["dielectric", "polarization"],
        default=None,
        help="Only for radnet",
    )
    parser.add_argument(
        "--n_equil",
        type=int,
        default=0,
        help="Number of time the equilibrium structure is added to the dataset.",
    )
    parser.add_argument(
        "--n_relaxed",
        type=int,
        default=0,
        help="Number of time the relaxation structures are added to the dataset.",
    )
    return parser


def main(args):
    if args.run_mode in ["bigdft", "abinit", "md"]:
        with connect(args.dbname) as db:
            if args.run_mode == "bigdft":
                db.metadata = DEFAULT_METADATA
                files = [f for f in os.listdir() if f.endswith(".xyz")]
                for f in files:
                    atoms = posinp_to_ase_atoms(Posinp.from_file(f))
                    with open(f, "r") as posinp_file:
                        energy = float(posinp_file.readline().split()[2]) * 27.21138602
                        forces = None
                        for line in posinp_file:
                            if "forces" not in line:
                                continue
                            else:
                                forces = []
                                for _ in range(len(atoms)):
                                    forces.append(
                                        [
                                            float(force)
                                            for force in posinp_file.readline().split()[
                                                -3:
                                            ]
                                        ]
                                    )
                                forces = np.array(forces) * 27.21138602 / 0.529177249
                                break
                    db.write(atoms, data={"energy": energy, "forces": forces})

            elif args.run_mode == "abinit":
                from abipy.abio.outputs import AbinitOutputFile

                db.metadata = DEFAULT_METADATA
                files = [
                    f for f in os.listdir() if f.endswith(".out") or f.endswith(".abo")
                ]
                if args.n_equil > 0:
                    if os.path.exists("equil.out"):
                        for _ in range(args.n_equil - 1):
                            files.append("equil.out")
                        random.shuffle(files)
                    else:
                        raise FileNotFoundError(
                            "The equilibrium positions should be in 'equil.out'."
                        )
                if args.n_relaxed > 0:
                    relaxed_files = [
                        f
                        for f in os.listdir()
                        if f.startswith("relax") and f.endswith(".xyz")
                    ]
                    if len(relaxed_files) == 0:
                        raise FileNotFoundError(
                            "The relaxation positions should be given in relax_X.xyz files."
                        )
                    else:
                        for rf in relaxed_files:
                            atoms = read(rf, format="extxyz")
                            energy = atoms.info["energy"]
                            forces = atoms.info["forces"]
                            for _ in range(args.n_relaxed):
                                db.write(
                                    atoms, data={"energy": energy, "forces": forces}
                                )

                for f in files:
                    atoms = read(f, format="abinit-out")
                    about = AbinitOutputFile(f)
                    energy = float(about.final_vars_global["etotal"]) * 27.21138602
                    forces = (
                        np.array(
                            [float(g) for g in about.final_vars_global["fcart"].split()]
                        ).reshape(-1, 3)
                        * 27.21138602
                        / 0.529177249
                    )
                    db.write(atoms, data={"energy": energy, "forces": forces})

            elif args.run_mode == "md":
                db.metadata = DEFAULT_MD_METADATA
                files = [f for f in os.listdir() if f.endswith(".xyz")]
                for i, f in enumerate(files):
                    atoms = read(f, format="extxyz")
                    energy = atoms.get_total_energy()
                    forces = atoms.get_forces()
                    db.write(atoms, data={"energy": energy, "forces": forces})

    elif args.run_mode in ["radnet"]:
        assert args.target is not None
        from abipy.abio.outputs import AbinitOutputFile
        import h5py

        h5FileName = args.dbname if args.dbname.endswith(".h5") else args.dbname + ".h5"
        with h5py.File(h5FileName, "w") as outfile:
            files = sorted(
                [f for f in os.listdir() if f.endswith(".out") or f.endswith(".abo")]
            )

            # Adding equilibrium structure multiple times
            if args.n_equil > 0:
                if os.path.exists("equil.out"):
                    for _ in range(args.n_equil - 1):
                        files.append("equil.out")
                    random.shuffle(files)  # Useful if we split the database later on
                else:
                    raise FileNotFoundError(
                        "The equilibrium positions should be in 'equil.out'."
                    )

            # Get common values between structures
            ref = read(files[0], format="abinit-out")
            natoms = len(ref)
            atomic_numbers = ref.get_atomic_numbers()
            cell = ref.cell
            au_cell = cell / Bohr
            au_volume = cell.volume / (Bohr**3)

            # Prepare arrays
            coordinates = np.empty((len(files), natoms, 3))
            dielectric = np.empty((len(files), 3, 3))
            polarization = np.empty((len(files), 3))

            # Loop on output files
            for i, f in enumerate(files):
                atoms = read(f, format="abinit-out")
                coordinates[i] = atoms.get_positions()

                about = AbinitOutputFile(f)

                # Read dielectric values
                die_data = about.datasets[3].split("\n")
                for j, line in enumerate(die_data):
                    if "Dielectric" in line:
                        die_results_idx = j

                die_values = []
                for offset in [4, 5, 6, 8, 9, 10, 12, 13, 14]:
                    die_values.append(die_data[die_results_idx + offset].split()[4])
                dielectric[i] = np.array(die_values).reshape(3, 3)

                # Read polarization values
                pol_data = about.datasets[4].split("\n")
                idx_list = []
                for j, line in enumerate(pol_data):
                    if "Electronic Berry phase" in line:
                        idx_list.append(j)

                # Unfold Berry phase to get continuous distribution
                pol_values = []
                for idx in idx_list:
                    p_elec = float(pol_data[idx].split()[3])
                    p_ion = float(pol_data[idx + 1].split()[2])
                    p_ion = 2 + p_ion if p_ion < 0 else p_ion
                    pol_values.append((p_elec + p_ion))
                pol_values = np.broadcast_to(np.array(pol_values), (3, 3)).T
                polarization[i] = (pol_values * au_cell).sum(0) / au_volume

            # Save targets
            if args.target == "polarization":
                target = polarization
            elif args.target == "dielectric":
                target = np.empty((len(files), 6))
                target[:, 0] = dielectric[:, 0, 0]
                target[:, 1] = dielectric[:, 0, 1]
                target[:, 2] = dielectric[:, 0, 2]
                target[:, 3] = dielectric[:, 1, 1]
                target[:, 4] = dielectric[:, 1, 2]
                target[:, 5] = dielectric[:, 2, 2]

            # Final write
            group = outfile.create_group(args.h5group)
            group.create_dataset("atomic_numbers", data=atomic_numbers)
            group.create_dataset("cell", data=cell.array)
            group.create_dataset("coordinates", data=coordinates)
            group.create_dataset("dielectric", data=dielectric)
            group.create_dataset("polarization", data=polarization)
            group.create_dataset("target", data=target)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
