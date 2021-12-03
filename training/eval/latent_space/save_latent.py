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
    name = args.dbpath.split("/")[-1] + "_ls" if args.name is None else args.name
    name = name if name.endswith(".pt") else name + ".pt"

    with connect(args.dbpath) as db:
        atoms = [row.toatoms() for row in db.select()]
    batch = get_latent_space_representations(model, atoms, output_rep=args.output)

    if args.output == False:
        representations = batch["representation"].cpu()
    else:
        representations = batch["output"].cpu()
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
    parser.add_argument("--cuda", action="store_true", help="Whether to use cuda.")
    parser.add_argument(
        "--output",
        action="store_true",
        help="Return the output module's latent space instead of the representation block's.",
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
