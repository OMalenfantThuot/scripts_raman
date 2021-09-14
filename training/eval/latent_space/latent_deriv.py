#!/usr/bin/env python

from utils.vibrations import GrapheneDDB
from schnetpack.utils import load_model
from schnetpack.environment import AseEnvironmentProvider
from mlcalcdriver.interfaces import SchnetPackData
from utils.latentspace import get_latent_space_representations
from schnetpack import AtomsLoader
import numpy as np
import torch
import pickle
import argparse


def main(args):
    ddb = GrapheneDDB(args.filepath)
    with open(args.dft_phonons, "rb") as f:
        dftph = pickle.load(f)
    with open(args.ml_phonons, "rb") as f:
        mlph = pickle.load(f)
    assert (
        dftph.path.kpts == mlph.path.kpts
    ).all(), "The DFT and ML kpts grids do not match."

    device = "cuda" if args.cuda else "cpu"
    model = load_model(args.modelpath, map_location=device)
    cutoff = float(model.representation.interactions[0].cutoff_network.cutoff)
    n_neurons = model.representation.n_atom_basis

    train_representations = torch.load(args.savepath).to(device)

    errorsph = np.abs(mlph.energies[0] - dftph.energies[0])
    latentdistances = np.zeros_like(errorsph)

    for i, qpoint in enumerate(mlph.path.kpts):
        for branch in range(6):
            if i % 100 == 0:
                print("qpoint #{}".format(i))
            displacements = ddb.build_supercell_modes(
                qpoint, branch, amplitudes=args.amp
            )
            representations = get_latent_space_representations(model, displacements)
            representations = representations.reshape(
                len(displacements[0]) * len(displacements), 1, n_neurons
            ).expand(
                len(displacements[0]) * len(displacements),
                train_representations.shape[0],
                n_neurons,
            )
            distances = (
                torch.linalg.norm(representations - train_representations, dim=2)
                .cpu()
                .detach()
                .numpy()
            )
            latentdistances[i, branch] = np.mean(
                np.sort(distances)[:, : args.n_neighbors]
            )

    savename = (
        args.savename if args.savename.endswith(".pkl") else args.savename + ".pkl"
    )
    results = {"errorsph": errorsph, "latentdistances": latentdistances}
    with open(savename, "wb") as f:
        pickle.dump(results, f)


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath", help="Path to the DDB file.",
    )
    parser.add_argument(
        "modelpath",
        help="Path to the model to get the latent space representations from.",
    )
    parser.add_argument(
        "savepath",
        help="Path to the .pt file containing the latent space representations of the training set.",
    )
    parser.add_argument(
        "dft_phonons",
        help="Path to the .pkl file containing the DFT phonon frequencies.",
    )
    parser.add_argument(
        "ml_phonons",
        help="Path to the .pkl file containing the ML phonons frequencies.",
    )
    parser.add_argument(
        "n_neighbors", help="Number of neighbors to consider.", type=int
    )
    parser.add_argument("savename", help="Name of the file to save the results.")
    parser.add_argument("--cuda", action="store_true", help="Wether to use cuda.")
    parser.add_argument(
        "--amp",
        nargs="*",
        help="""
            Amplitudes of the perturbations. If multiples values given, will create
            multiple perturbations and return the mean error.
            """,
        type=float,
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
