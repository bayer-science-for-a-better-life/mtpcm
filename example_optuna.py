# pylint: disable=import-error

from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from optuna.samplers import TPESampler
from argparse import Namespace
import argparse
from PCM.optuna import (
    Objective_ST,
    Objective_ST_ext,
    Objective_MT,
    Objective_MT_withPRT,
)
from pytorch_lightning import seed_everything


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Run Optune to determine optimal model HPs"
    )
    parser.add_argument("--model_dir", type=str,
                        help="Model directory", required=True)
    parser.add_argument(
        "--method",
        type=str,
        help="Which approach to use",
        required=True,
    )
    parser.add_argument("--censored", action="store_true")
    parser.add_argument("--batch_size", type=int,
                        help="Batch size", default=512)
    parser.add_argument(
        "--noise",
        type=float,
        help="Noise level applied to cmp and prt at training time",
        default=0.05,
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = vars(arg_parser())
    print(args)

    # prepare data
    model_dir = Path(args["model_dir"])
    optuna_dir = Path(model_dir / "OptunaHPSearch")
    par = {
        "censored": args["censored"],
        "batch_size": args["batch_size"],
        "noise": args["noise"],
    }

    # Get the data
    N_train = 1000
    N_val = 100

    cmp_tr = np.random.rand(N_train, 512) * 2 - 1.0
    prt_tr = np.random.rand(N_train, 256)
    pIC50_tr = np.random.rand(N_train) + 5
    prefixes_tr = np.random.randint(low=-1, high=2, size=N_train)

    cmp_val = np.random.rand(N_val, 512) * 2 - 1.0
    prt_val = np.random.rand(N_val, 256)
    pIC50_val = np.random.rand(N_val) + 5
    prefixes_val = np.random.randint(low=-1, high=2, size=N_val)

    if args["method"] == "PCM":

        asy_tr = np.vstack([[0, 1], [1, 0]] * int(N_train / 2))
        asy_val = np.vstack([[0, 1], [1, 0]] * int(N_val / 2))

        data_params = Namespace(**par)
        data_train = {
            "prt": prt_tr,
            "cmp": cmp_tr,
            "asy": asy_tr,
            "pIC50": pIC50_tr,
            "prefixes": prefixes_tr,
        }
        data_val = {
            "prt": prt_val,
            "cmp": cmp_val,
            "asy": asy_val,
            "pIC50": pIC50_val,
            "prefixes": prefixes_val,
        }
        objective = Objective_ST(
            optuna_dir, data_params, data_train, data_val, data_test=None
        )

    elif args["method"] == "PCM_ext":

        par["num_tasks"] = 5
        data_params = Namespace(**par)
        asy_tr = np.array([0, 1, 2, 3, 4] * int(N_train / 5)).astype(int)
        asy_val = np.array([0, 1, 2, 3, 4] * int(N_val / 5)).astype(int)

        data_train = {
            "prt": prt_tr,
            "cmp": cmp_tr,
            "asy": asy_tr,
            "pIC50": pIC50_tr,
            "prefixes": prefixes_tr,
        }
        data_val = {
            "prt": prt_val,
            "cmp": cmp_val,
            "asy": asy_val,
            "pIC50": pIC50_val,
            "prefixes": prefixes_val,
        }
        objective = Objective_ST_ext(
            optuna_dir, data_params, data_train, data_val, data_test=None
        )

    elif args["method"] == "PCM_MT":

        par["num_tasks"] = 5
        taskind_tr = np.array([0, 1, 2, 3, 4] * int(N_train / 5)).astype(int)
        taskind_val = np.array([0, 1, 2, 3, 4] * int(N_val / 5)).astype(int)

        data_params = Namespace(**par)
        data_train = {
            "cmp": cmp_tr,
            "pIC50": pIC50_tr,
            "prefixes": prefixes_tr,
            "taskind": taskind_tr,
        }
        data_val = {
            "cmp": cmp_val,
            "pIC50": pIC50_val,
            "prefixes": prefixes_val,
            "taskind": taskind_val,
        }
        objective = Objective_MT(
            optuna_dir, data_params, data_train, data_val, data_test=None
        )

    elif args["method"] == "PCM_MT_withPRT":

        par["num_tasks"] = 5
        taskind_tr = np.array([0, 1, 2, 3, 4] * int(N_train / 5)).astype(int)
        taskind_val = np.array([0, 1, 2, 3, 4] * int(N_val / 5)).astype(int)

        data_params = Namespace(**par)
        data_train = {
            "prt": prt_tr,
            "cmp": cmp_tr,
            "pIC50": pIC50_tr,
            "prefixes": prefixes_tr,
            "taskind": taskind_tr,
        }
        data_val = {
            "prt": prt_val,
            "cmp": cmp_val,
            "pIC50": pIC50_val,
            "prefixes": prefixes_val,
            "taskind": taskind_val,
        }
        objective = Objective_MT_withPRT(
            optuna_dir, data_params, data_train, data_val, data_test=None
        )

    seed_everything(42, workers=True)
    study_name = args["method"]
    sampler = TPESampler(seed=10)
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///%s/%s.db" % (str(optuna_dir), study_name),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
    )

    study.optimize(objective, n_trials=2)

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
