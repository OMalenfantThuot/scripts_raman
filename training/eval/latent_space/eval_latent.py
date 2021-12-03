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
from schnetpack.nn import shifted_softplus

r"""
This executable needs a saved representation from 'save_latent.py'.
It calculates the representations of a test set and returns the distances
of each atom of each test configuration compared to the training set.
This distance can be dependent on a certain numebr of nearest neighbors
(lower is closer) or a gaussian weighted sum on all points of the training set
(larger is closer).
"""


def main(args):

    if args.intermediate and args.output:
        raise ValueError(
            """
            The arguments intermediate and output
            cannot be used at the same time.
            """
        )

    device = "cuda" if args.cuda else "cpu"
    model = load_model(args.modelpath, map_location=device)
    n_neurons = model.representation.n_atom_basis

    if args.output:
        from utils.models import LatentAtomwise

        latent_out = LatentAtomwise.from_Atomwise(model.output_modules[0])
        latent_out.to(next(model.parameters()).device)
        n_neurons = int(n_neurons / 2)

    if args.intermediate:
        model.representation.return_intermediate = True

    train_representations = torch.load(args.savepath).to(device)
    if args.output:
        train_representations = shifted_softplus(train_representations)

    with connect(args.dbpath) as db:
        n_struct = db.count()
        atoms = [row.toatoms() for row in db.select()]
        natoms = [len(atom) for atom in atoms]

    results = {}
    for i, atom in enumerate(atoms):
        results[i] = {}
        inputs = get_latent_space_representations(
            model,
            atom,
            output_rep=args.output,
            latent_out=latent_out if args.output else None,
        )
        factors = get_scaling_factors(
            inputs, train_representations, scaling=args.scaling, model=model
        ).to(device)

        if args.intermediate:
            representations = inputs["representation"][1]
        elif args.output:
            representations = [inputs["output"]]
            representations[0] = shifted_softplus(representations[0])
        else:
            representations = [inputs["representation"]]

        for j, rep in enumerate(representations):
            rep = rep.reshape(natoms[i], 1, n_neurons).expand(
                natoms[i], train_representations.shape[0], n_neurons
            )
            distances = get_latent_space_distances(
                rep,
                train_representations,
                factors,
                metric=args.metric,
                grad=args.scaling == "gradient",
            )

            if args.distances_mode == "neighbors":
                distances = np.mean(np.sort(distances)[:, : args.n_neighbors], axis=1)
            elif args.distances_mode == "gaussians":
                distances = np.float64(distances)
                distances = np.sum(np.exp(-distances / (2 * args.std)), axis=1)

            results[i][j] = distances

    name = args.dbpath.split("/")[-1] if args.name is None else args.name
    name = name + ".pkl" if not name.endswith(".pkl") else name
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
        "--scaling",
        choices=["isotropic", "scaled_max", "scaled_std", "gradient"],
        help="How to scale the different dimensions of the latent space",
        default="isotropic",
    )
    parser.add_argument(
        "--metric",
        choices=["euclidian", "linear"],
        help="The metric used to calculate the norm of the dsitances.",
        default="euclidian",
    )
    parser.add_argument(
        "--distances_mode",
        choices=["neighbors", "gaussians"],
        default="neighbors",
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
    parser.add_argument(
        "--intermediate",
        action="store_true",
        help="Use to store the intermediate representations.",
        default=False,
    )
    parser.add_argument(
        "--output",
        action="store_true",
        help="""
            Get the latent space representation in the output
            module instead of the representation.
            """,
        default=False,
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
