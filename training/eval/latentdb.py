from ase.db import connect
from schnetpack.utils import load_model
from schnetpac
import argparse
import torch
import os


def main(args):

    device = "cuda" if args.cuda else "cpu"
    model = load_model(args.modelpath, map_location=device)
    n_neurons = model.representation.n_atom_basis

    with connect(args.dbpath) as db:
        n_struct = db.count()

    pass


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
