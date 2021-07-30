#!/usr/bin/env python

from ase.db import connect
from schnetpack.utils import load_model
from schnetpack.environment import AseEnvironmentProvider
from mlcalcdriver.interfaces import SchnetPackData
from schnetpack import AtomsLoader
import argparse
import torch
import os


def main(args):

    device = "cuda" if args.cuda else "cpu"
    model = load_model(args.modelpath, map_location=device)
    cutoff = float(model.representation.interactions[0].cutoff_network.cutoff)
    n_neurons = model.representation.n_atom_basis
    individual_reps = []

    with connect(args.dbpath) as db:
        n_struct = db.count()
        atoms = [row.toatoms() for row in db.select()]
    data = SchnetPackData(
        data=atoms,
        environment_provider=AseEnvironmentProvider(cutoff=cutoff),
        collect_triples=False,
    )
    data_loader = AtomsLoader(data, batch_size=1)

    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        rep = model.representation(batch)
        individual_reps.append(rep)
    representations = torch.cat(individual_reps).reshape(-1, n_neurons)

    name = args.dbpath.split("/")[-1] + "_ls" if args.name is None else args.name
    name = name + ".pt"
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
