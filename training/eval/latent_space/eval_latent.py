#!/usr/bin/env python

from ase.db import connect
from schnetpack.utils import load_model
from utils.latentspace import (
    get_latent_space_representations,
    get_latent_space_distances,
    get_scaling_factors,
)
import numpy as np
import pickle
import torch
import argparse

r"""
This executable needs a saved representation from 'save_latent.py'.
It calculates the representations of a test set and returns the distances
of each atom of each test configuration compared to the training set.
This distance can be dependent on a certain numebr of nearest neighbors
(lower is closer) or a gaussian weighted sum on all points of the training set
(larger is closer).
"""


def main(args):

    device = "cuda" if args.cuda else "cpu"
    model = load_model(args.modelpath, map_location=device)
    n_neurons = model.representation.n_atom_basis

    train_representations = torch.load(args.savepath).to(device)
    results = {}

    with connect(args.dbpath) as db:
        n_struct = db.count()
        atoms = [row.toatoms() for row in db.select()]
        natoms = [len(atom) for atom in atoms]

    for i, atom in enumerate(atoms):
        inputs = get_latent_space_representations(model, atom)
        factors = get_scaling_factors(
            inputs, train_representations, metric=args.metric, model=model
        ).to(device)
        representations = inputs["representation"]
        representations = representations.reshape(natoms[i], 1, n_neurons).expand(
            natoms[i], train_representations.shape[0], n_neurons
        )
        distances = get_latent_space_distances(
            representations, train_representations, factors, metric=args.metric,
        )
        if args.distances_mode == "neighbors":
            distances = np.mean(np.sort(distances)[:, : args.n_neighbors], axis=1)
        elif args.distances_mode == "gaussians":
            distances = np.float64(distances)
            distances = np.sum(np.exp(-distances / (2 * args.std)), axis=1)
        results[i] = distances

    name = (
        args.name + ".pkl"
        if args.name is not None
        else args.dbpath.split("/")[-1] + ".pkl"
    )
    with open(name, "wb") as f:
        pickle.dump(results, f)


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "dbpath", help="Path to the database to get the latent space distances for.",
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
        "--name", help="Name of the file to save the distances in (optional)."
    )
    parser.add_argument("--cuda", action="store_true", help="Wether to use cuda.")
    parser.add_argument(
        "--metric",
        choices=["euclidian", "scaled_max", "scaled_std", "gradient"],
        help="Metric to use to calculate the distance.",
        default="euclidian",
    )
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
        "--std",
        help="Standard deviation for the gaussian sum.",
        type=float,
        default=0.1,
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
