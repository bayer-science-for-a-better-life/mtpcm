# pylint: disable=import-error

from pathlib import Path
import pandas as pd
import numpy as np
from argparse import Namespace
import argparse
from PCM.PL_PCM import PCM, PCM_ext
from PCM.PL_MTL import PCM_MT, PCM_MT_withPRT
from PCM.datamodules import (
    PCM_DataModule_ST,
    PCM_DataModule_MT,
    PCM_DataModule_MT_withPRT,
)
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything


def arg_parser():
    parser = argparse.ArgumentParser(
        description="Minimal example of training a model on snthetic data."
    )
    parser.add_argument(
        "--method",
        type=str,
        help="Which model to use: PCM, PCM_ext, MT or MT_withPRT",
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

    par = {
        "censored": args["censored"],
        "batch_size": args["batch_size"],
        "noise": args["noise"],
        "lr": 0.0001,
        "num_layers": 1,
        "cmp_size": 128,
        "layers": [512],
        "dropout": 0.1,
    }

    # Get the data
    N_train = 1000
    N_test = 100
    N_val = 100

    cmp_tr = np.random.rand(N_train, 512) * 2 - 1.0
    prt_tr = np.random.rand(N_train, 256)
    pIC50_tr = np.random.rand(N_train) + 5
    prefixes_tr = np.random.randint(low=-1, high=2, size=N_train)

    cmp_val = np.random.rand(N_val, 512) * 2 - 1.0
    prt_val = np.random.rand(N_val, 256)
    pIC50_val = np.random.rand(N_val) + 5
    prefixes_val = np.random.randint(low=-1, high=2, size=N_val)

    cmp_test = np.random.rand(N_test, 512) * 2 - 1.0
    prt_test = np.random.rand(N_test, 256)
    pIC50_test = np.random.rand(N_test) + 5
    prefixes_test = np.random.randint(low=-1, high=2, size=N_test)

    if args["method"] == "PCM":

        asy_tr = np.vstack([[0, 1], [1, 0]] * int(N_train / 2))
        asy_val = np.vstack([[0, 1], [1, 0]] * int(N_val / 2))
        asy_test = np.vstack([[0, 1], [1, 0]] * int(N_test / 2))

        par["prt_size"] = 128

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
        data_test = {
            "prt": prt_test,
            "cmp": cmp_test,
            "asy": asy_test,
            "pIC50": pIC50_test,
            "prefixes": prefixes_test,
        }
        model = PCM(par)
        datamodule = PCM_DataModule_ST(
            data_train,
            data_val,
            data_test,
            censored=par["censored"],
            cmp_noise=par["noise"],
            prt_noise=par["noise"],
            batch_size=par["batch_size"],
        )

    elif args["method"] == "PCM_ext":

        par["prt_size"] = 128
        par["num_tasks"] = 5
        asy_tr = np.array([0, 1, 2, 3, 4] * int(N_train / 5)).astype(int)
        asy_val = np.array([0, 1, 2, 3, 4] * int(N_val / 5)).astype(int)
        asy_test = np.array([0, 1, 2, 3, 4] * int(N_test / 5)).astype(int)

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
        data_test = {
            "prt": prt_test,
            "cmp": cmp_test,
            "asy": asy_test,
            "pIC50": pIC50_test,
            "prefixes": prefixes_test,
        }
        model = PCM_ext(par)
        datamodule = PCM_DataModule_ST(
            data_train,
            data_val,
            data_test,
            censored=par["censored"],
            cmp_noise=par["noise"],
            prt_noise=par["noise"],
            batch_size=par["batch_size"],
            norm_asy=False
        )

    elif args["method"] == "PCM_MT":

        par["num_tasks"] = 5
        taskind_tr = np.array([0, 1, 2, 3, 4] * int(N_train / 5)).astype(int)
        taskind_val = np.array([0, 1, 2, 3, 4] * int(N_val / 5)).astype(int)
        taskind_test = np.array([0, 1, 2, 3, 4] * int(N_test / 5)).astype(int)

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
        data_test = {
            "cmp": cmp_test,
            "pIC50": pIC50_test,
            "prefixes": prefixes_test,
            "taskind": taskind_test,
        }
        model = PCM_MT(par)
        datamodule = PCM_DataModule_MT(
            par["num_tasks"],
            data_train,
            data_val,
            data_test,
            censored=par["censored"],
            cmp_noise=par["noise"],
            batch_size=par["batch_size"]
        )

    elif args["method"] == "PCM_MT_withPRT":

        par["num_tasks"] = 5
        par["prt_size"] = 128
        taskind_tr = np.array([0, 1, 2, 3, 4] * int(N_train / 5)).astype(int)
        taskind_val = np.array([0, 1, 2, 3, 4] * int(N_val / 5)).astype(int)
        taskind_test = np.array([0, 1, 2, 3, 4] * int(N_test / 5)).astype(int)

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
        data_test = {
            "prt": prt_test,
            "cmp": cmp_test,
            "pIC50": pIC50_test,
            "prefixes": prefixes_test,
            "taskind": taskind_test,
        }
        model = PCM_MT_withPRT(par)
        datamodule = PCM_DataModule_MT_withPRT(
            par["num_tasks"],
            data_train,
            data_val,
            data_test,
            censored=par["censored"],
            cmp_noise=par["noise"],
            prt_noise=par["noise"],
            batch_size=par["batch_size"]
        )

    trainer = pl.Trainer(
        logger=False,
        max_epochs=5,
        gpus=1,
        progress_bar_refresh_rate=0,
        deterministic=True,
    )
    print("Run training")
    trainer.fit(model, datamodule)

    print("Run testing")
    trainer.test(model, datamodule)
