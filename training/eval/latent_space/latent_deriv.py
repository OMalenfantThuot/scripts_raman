#!/usr/bin/env python

from utils.vibrations import GrapheneDDB
from schnetpack.utils import load_model
from utils.latentspace import get_latent_space_representations
import numpy as np
import torch
import pickle
import argparse

r"""
This executable creates displacements in a graphene supercell
equivalent to a certain qpoint and branch from an Abinit DDB file.
It can then get the latent representation of this displacement.
It then compares the DFT values to ML values to get the corresponding
ML errors and saves both values so that they can be plotted easily after.
"""


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
    memory_ok = True

    for i, qpoint in enumerate(mlph.path.kpts):
        for branch in range(6):
            displacements = ddb.build_supercell_modes(
                qpoint, branch, amplitudes=args.amp
            )
            representations = get_latent_space_representations(model, displacements)
            if memory_ok:
                try:
                    representations = representations.reshape(
                        len(displacements[0]) * len(displacements), 1, n_neurons
                    ).expand(
                        len(displacements[0]) * len(displacements),
                        train_representations.shape[0],
                        n_neurons,
                    )
                    distances = (
                        torch.linalg.norm(
                            representations - train_representations, dim=2
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )
                except (RuntimeError):
                    memory_ok = False
                    distances = loop_distances(representations, train_representations)
            else:
                distances = loop_distances(representations, train_representations)

            if args.distances_mode == "neighbors":
                if not args.scaled:
                    latentdistances[i, branch] = np.mean(
                        np.sort(distances)[:, : args.n_neighbors],
                    )
                else:
                    raise NotImplementedError()
                    if args.scaling == "linear":
                        # Donne plus d'importance aux atomes plus loin du training set
                        # Ce bloc est à tester

                        latentdistances[i, branch] = np.dot(
                            np.sort(
                                np.mean(
                                    np.sort(distances)[:, :, args.n_neighbors], axis=1
                                )
                            ),
                            np.linspace(0, 1, distances.shape[0]),
                        )

            elif args.distances_mode == "gaussians":
                distances = np.float64(distances)
                distances = np.sort(
                    np.log10(np.sum(np.exp(-distances / (2 * args.std)), axis=1))
                )
                latentdistances[i, branch] = np.dot(
                    distances, np.linspace(1, 0, distances.shape[0])
                )

    savename = (
        args.savename if args.savename.endswith(".pkl") else args.savename + ".pkl"
    )
    results = {"errorsph": errorsph, "latentdistances": latentdistances}
    with open(savename, "wb") as f:
        pickle.dump(results, f)


def loop_distances(representations, train_representations):
    distances = np.zeros((representations.shape[0], train_representations.shape[0]))
    for i, atom in enumerate(representations):
        distances[i] = (
            torch.linalg.norm(atom - train_representations, dim=1)
            .cpu()
            .detach()
            .numpy()
        )
    return distances


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
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
    parser.add_argument("savename", help="Name of the file to save the results.")
    parser.add_argument("--cuda", action="store_true", help="Wether to use cuda.")
    parser.add_argument(
        "--distances_mode",
        choices=["neighbors", "gaussians"],
        default="gaussians",
        help="""
            Distance can be either mean of the nearest neighbors
            or the gaussian sum of all the training points.
            """,
    )
    parser.add_argument(
        "--n_neighbors",
        help="Number of neighbors to consider (only needed for the neighbors distances).",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--scaled",
        help="Scale the distances of the neighbors.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--std",
        help="Standard deviation for the gaussian sum.",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--amp",
        nargs="*",
        help="""
            Amplitudes of the perturbations. If multiples values given, will create
            multiple perturbations and return the mean error.
            """,
        type=float,
        default=0.1,
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
