#! /usr/bin/env python

from train import new_training
from eval import eval_last_step
import argparse
import os


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    restart_subparser = parser.add_subparsers(
        dest="start_mode", help="Start new training or restart to next step."
    )

    start_parser = restart_subparser.add_parser(
        "start",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Start new training.",
    )
    start_parser.add_argument(
        "args_file",
        help="Name of the training args file (without the database and size args).",
    )
    start_parser.add_argument(
        "submission_file",
        help="Name of the training submission file.",
    )
    start_parser.add_argument(
        "n_train", help="Number of training repetitions.", type=int, default=10
    )
    start_parser.add_argument("original_data", help="Path to the original dataset.")

    restart_parser = restart_subparser.add_parser(
        "restart",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help="Restart training",
    )
    restart_parser.add_argument("restart_file", help="Path to the restart file.")
    restart_parser.add_argument(
        "--force_evaluation",
        default=False,
        action="store_true",
        help="Force to do the evaluation even if all trainings are not complete.",
    )
    return parser


def main(args):
    if args.start_mode == "start":
        new_training(args)
    elif args.start_mode == "restart":
        eval_last_step(args)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
