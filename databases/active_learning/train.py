import os
import shutil
from schnetpack.utils import read_from_json, to_json
from ase.db import connect
import yaml


def new_training(args):

    cwd = os.getcwd()
    original_db_path = os.path.join(cwd, args.original_data)
    with connect(original_db_path) as db:
        n_structs = len(db)
    args_path = os.path.join(cwd, args.args_file)
    submission_path = os.path.join(cwd, args.submission_file)
    train_args = read_from_json(args_path)
    train_args.__dict__.update(
        {
            "datapath": original_db_path,
            "split": [int(0.8 * n_structs), int(0.2 * n_structs)],
        }
    )

    os.makedirs("step_00/", exist_ok=True)
    os.chdir("step_00/")

    os.makedirs("train_dir/")
    os.chdir("train_dir/")

    for i in range(1, args.n_train + 1):
        train_dir = "train_{:02}".format(i)
        os.makedirs(os.path.join(train_dir, "model/"))
        shutil.copy(submission_path, train_dir)
        to_json(os.path.join(train_dir, "model/args.json"), train_args.__dict__)

        os.chdir(train_dir)
        os.system("sbatch submit.sh")
        os.chdir("../")

    os.chdir(cwd)
    restart_yaml = {"current_step": 0, 0: {"n_train": args.n_train, "train_complete": False}}
    with open("restart_file.yaml", "w") as f:
        yaml.dump(restart_yaml, f)
