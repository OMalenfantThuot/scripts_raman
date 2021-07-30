#!/usr/bin/env python

from ase.db import connect
from schnetpack.utils import load_model
from schnetpack.environment import AseEnvironmentProvider
from mlcalcdriver.interfaces import SchnetPackData
from schnetpack import AtomsLoader
import pickle
import torch
import argparse
import torch
import os


def main(args):

    device = "cuda" if args.cuda else "cpu"
    model = load_model(args.modelpath, map_location=device)
    cutoff = float(model.representation.interactions[0].cutoff_network.cutoff)
    n_neurons = model.representation.n_atom_basis

    train_representations = torch.load(args.savepath)
    results = {}

    with connect(args.dbpath) as db:
        n_struct = db.count()
        atoms = [row.toatoms() for row in db.select()]
        natoms = [len(atom) for atom in atoms]
    data = SchnetPackData(
        data=atoms,
        environment_provider=AseEnvironmentProvider(cutoff=cutoff),
        collect_triples=False,
    )
    data_loader = AtomsLoader(data, batch_size=1)

    for i, batch in enumerate(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            rep = (
                model.representation(batch)
                .reshape(natoms[i], 1, n_neurons)
                .expand(natoms[i], train_representations.shape[0], n_neurons)
            )
        distances = torch.linalg.norm(rep - train_representations, dim=2)
        results[i] = distances.cpu().detach().numpy()

    name = (
        args.name + ".pkl"
        if args.name is not None
        else args.dbpath.split("/")[-1] + ".pkl"
    )
    with open(name, "wb") as f:
        pickle.dump(results, f)


def create_parser():
    parser = argparse.ArgumentParser()
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
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
