#!/usr/bin/env python

from ase.db import connect
from schnetpack.utils import load_model
from utils.latentspace import get_latent_space_representations
import argparse
import torch
import os

r"""
This executable saves the latent space representations
of all configurations in a dataset for a given trained model.
They are saved as a (n_configurations, n_neurons) Tensor
in a .pt file.
"""


def main(args):

    device = "cuda" if args.cuda else "cpu"
    model = load_model(args.modelpath, map_location=device)

    with connect(args.dbpath) as db:
        atoms = [row.toatoms() for row in db.select()]
    representations = get_latent_space_representations(model, atoms)["representation"][
        0
    ].cpu()

    name = args.dbpath.split("/")[-1] + "_ls" if args.name is None else args.name
    name = name if name.endswith(".pt") else name + ".pt"
    torch.save(representations, name)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dbpath",
        help="Path to the database to get the latent space representations from.",
    )
    parser.add_argument(
        "modelpath",
        help="Path to the model to get the latent space representations from.",
    )
    parser.add_argument(
        "--name", help="Name of the file to save the Tensors in (optional)."
    )
    parser.add_argument("--cuda", action="store_true", help="Wether to use cuda.")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
