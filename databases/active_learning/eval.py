import os
import yaml
import shutil


def eval_last_step(args):
    with open(args.restart_file, "r") as f:
        restart_dict = yaml.load(f, Loader=yaml.FullLoader)

    step = restart_dict["current_step"]
    step_dir = "step_{:02}".format(step)
    train_dir = os.path.join(step_dir, "train_dir")
    eval_dir = os.path.join(step_dir, "eval_dir")
    os.makedirs(eval_dir, exist_ok=True)

    n_train = restart_dict[step]["n_train"]
    do_evaluation = True

    for i in range(1, n_train + 1):
        model_dir = os.path.join(train_dir, "train_{:02}".format(i), "model/")

        log_path = os.path.join(model_dir, "log", "log.csv")
        with open(log_path, "r") as f:
            final_lr = f.readlines()[-1].split(",")[1]

        if final_lr == "1e-06":
            shutil.copy(
                os.path.join(model_dir, "best_model"),
                os.path.join(eval_dir, "model_{:02}".format(i)),
            )
        else:
            print(
                "Training of model {:02} in step {:02} is not complete.".format(i, step)
            )
            do_evaluation = False

    if args.force_evaluation:
        do_evaluation = True

    if do_evaluation:
        restart_dict[step]["training_complete"] = True
        os.chdir(eval_dir)

        models = [f for f in os.listdir() if f.startswith("model")]
        for model in models:
            print(model)
