import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_type", help="Type of model to train.")
    parser.add_argument("ndata", type=int, help="Number of data points to generate.")
    parser.add_argument(
        "--function",
        default="cos",
        choices=["cos", "sin", "lj", "quad", "p4"],
        help="Analytic function to learn.",
    )
    parser.add_argument(
        "--range",
        default=[0.0, 1.0],
        type=float,
        help="Range for the interatomic distance for which data will be generated.",
        nargs=2,
    )
    parser.add_argument(
        "--cuda", default=False, action="store_true", help="Use GPU for training."
    )
    return parser
