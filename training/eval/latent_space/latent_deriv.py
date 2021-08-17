from phonon_projections.utils.vibrations import GrapheneDDB
import pickle
import argparse


def main(args):
    ddb = GrapheneDDB(args.filepath)
    with open(args.dft_phonons, "rb") as f:
        dftph = pickle.load(f)
    with open(args.ml_phonons, "rb") as f:
        mlph = pickle.load(f)
    pass


def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filepath",
        help="Path to the DDB file.",
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
        help="Path to the .pkl file containing the ML phonons freuqencies.",
    )
    parser.add_argument(
        "n_neighbors", help="Number of neighbors to consider.", type=int
    )
    parser.add_argument("savename", help="Name of the file to save the results.")
    parser.add_argument("--cuda", action="store_true", help="Wether to use cuda.")
    parser.add_argument(
        "--amp",
        nargs="*",
        help="Amplitudes of the perturbations. If multiples values given, will create multiple perturbations and return the mean error.",
        type=float,
    )
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
