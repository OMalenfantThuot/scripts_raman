#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import os


def import_mol(data_dict, molecule):
    data_dict[molecule] = mol_dict(molecule)


def mol_dict(molecule):
    mol_dict = {}
    for run in os.listdir("data/" + molecule):
        if run == "gnrm":
            for subrun in os.listdir("data/" + molecule + "/gnrm/"):
                mol_dict["gnrm_" + subrun] = np.load(
                    "data/"
                    + molecule
                    + "/"
                    + run
                    + "/"
                    + subrun
                    + "/"
                    + molecule
                    + "_ph_energies.npy"
                )
        else:
            mol_dict[run] = np.load(
                "data/" + molecule + "/" + run + "/" + molecule + "_ph_energies.npy"
            )
    return mol_dict


if __name__ == "__main__":

    rmult_list = [4.0, 5.0, 6.0, 7.0, 8.0]
    rmult_labels = ["(4, 7)", "(5, 8)", "(6, 9)", "(7, 10)", "(8, 11)"]
    hgrid_list = [0.48, 0.44, 0.40, 0.36, 0.32, 0.28, 0.24, 0.20, 0.16]

    data_dict = {}
    avail_molecules = os.listdir("data/")
    for mol in avail_molecules:
        import_mol(data_dict, mol)

    for molecule in data_dict.keys():
        energies = data_dict[molecule]["pseudos"]
        fig, ax = plt.subplots(1, energies.shape[0])
        fig.suptitle("Convergence " + molecule)
        for i in range(energies.shape[0]):
            for j in range(energies.shape[2]):
                ax[i].plot(hgrid_list, energies[i].T[j], "o")
            ax[i].invert_xaxis()
            ax[i].set_xlabel("hgrid")
            ax[i].set_title("rmult " + rmult_labels[i])
        plt.show()
