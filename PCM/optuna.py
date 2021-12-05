import torch
import numpy as np
import pytorch_lightning as pl
from optuna.integration import PyTorchLightningPruningCallback
from PCM.PL_PCM import PCM, PCM_ext
from PCM.PL_MTL import PCM_MT, PCM_MT_withPRT
from PCM.datamodules import (
    PCM_DataModule_ST,
    PCM_DataModule_MT,
    PCM_DataModule_MT_withPRT,
)
from pytorch_lightning import seed_everything
from PCM.utils import MetricsCallback


def sample_params(trial, prt=True):
    params = {}
    params["lr"] = 10 ** (trial.suggest_discrete_uniform("learning_rate", -5, -3, 1))
    if prt:
        params["prt_size"] = int(
            2 ** (trial.suggest_discrete_uniform("prt_bottleneck_size", 2, 10, 1))
        )
    params["cmp_size"] = int(
        2 ** (trial.suggest_discrete_uniform("cmp_bottleneck_size", 2, 10, 1))
    )
    params["num_layers"] = trial.suggest_int(
        "n_layers", 1, 5
    )  # trial.suggest_int("n_layers", 1, 6)
    params["dropout"] = trial.suggest_discrete_uniform(
        "dropout", 0.0, 0.5, 0.1)
    params["layers"] = []
    max_size = 10  # 12
    for i in range(params["num_layers"]):
        params["layers"].append(
            int(
                2
                ** (
                    trial.suggest_discrete_uniform(
                        "n_units_l{}".format(i), 2, max_size, 1
                    )
                )
            )
        )
        max_size = int(np.log2(params["layers"][-1]))
    return params


class Objective_ST(object):
    def __init__(self, model_dir, data_params, data_train, data_val, data_test=None):
        self.data_params = data_params
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.model_dir = model_dir

    def __call__(self, trial):

        trainer = pl.Trainer(
            logger=False,
            checkpoint_callback=False,
            num_sanity_val_steps=0,
            max_epochs=20,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(
                trial, monitor="val_r2")],
            progress_bar_refresh_rate=0,
            deterministic=True,
        )

        optuna_params = sample_params(trial)

        model = PCM(optuna_params)
        datamodule = PCM_DataModule_ST(
            self.data_train,
            self.data_val,
            self.data_test,
            censored=self.data_params.censored,
            cmp_noise=self.data_params.noise,
            prt_noise=self.data_params.noise,
            batch_size=self.data_params.batch_size,
        )
        trainer.fit(model, datamodule)
        # print(trainer.logged_metrics.keys())
        return trainer.callback_metrics["val_r2"].item()


class Objective_ST_ext(object):
    def __init__(self, model_dir, data_params, data_train, data_val, data_test=None):
        self.data_params = data_params
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.model_dir = model_dir

    def __call__(self, trial):

        trainer = pl.Trainer(
            logger=False,
            checkpoint_callback=False,
            num_sanity_val_steps=5,
            max_epochs=50,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(
                trial, monitor="val_r2")],
            progress_bar_refresh_rate=0,
            deterministic=True,
        )

        optuna_params = sample_params(trial)
        optuna_params["num_tasks"] = self.data_params.num_tasks
        print(optuna_params)

        model = PCM_ext(optuna_params)
        datamodule = PCM_DataModule_ST(
            self.data_train,
            self.data_val,
            self.data_test,
            censored=self.data_params.censored,
            cmp_noise=self.data_params.noise,
            prt_noise=self.data_params.noise,
            batch_size=self.data_params.batch_size,
            norm_asy=False,
        )
        trainer.fit(model, datamodule)

        return trainer.callback_metrics["val_r2"].item()


class Objective_MT(object):
    def __init__(self, model_dir, data_params, data_train, data_val, data_test=None):
        self.data_params = data_params
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.model_dir = model_dir

    def __call__(self, trial):

        trainer = pl.Trainer(
            logger=False,
            checkpoint_callback=False,
            num_sanity_val_steps=0,
            max_epochs=50,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(
                trial, monitor="val_r2")],
            progress_bar_refresh_rate=0,
            deterministic=True,
        )

        optuna_params = sample_params(trial, prt=False)
        optuna_params["num_tasks"] = self.data_params.num_tasks

        model = PCM_MT(optuna_params)
        datamodule = PCM_DataModule_MT(
            self.data_params.num_tasks,
            self.data_train,
            self.data_val,
            self.data_test,
            censored=self.data_params.censored,
            cmp_noise=self.data_params.noise,
            batch_size=self.data_params.batch_size,
        )
        trainer.fit(model, datamodule)

        return trainer.callback_metrics["val_r2"].item()


class Objective_MT_withPRT(object):
    def __init__(self, model_dir, data_params, data_train, data_val, data_test=None):
        self.data_params = data_params
        self.data_train = data_train
        self.data_test = data_test
        self.data_val = data_val
        self.model_dir = model_dir

    def __call__(self, trial):

        trainer = pl.Trainer(
            logger=False,
            checkpoint_callback=False,
            num_sanity_val_steps=0,
            max_epochs=50,
            gpus=1 if torch.cuda.is_available() else None,
            callbacks=[PyTorchLightningPruningCallback(
                trial, monitor="val_r2")],
            progress_bar_refresh_rate=0,
            deterministic=True,
        )

        optuna_params = sample_params(trial)
        optuna_params["num_tasks"] = self.data_params.num_tasks

        model = PCM_MT_withPRT(optuna_params)
        datamodule = PCM_DataModule_MT_withPRT(
            self.data_params.num_tasks,
            self.data_train,
            self.data_val,
            self.data_test,
            censored=self.data_params.censored,
            cmp_noise=self.data_params.noise,
            prt_noise=self.data_params.noise,
            batch_size=self.data_params.batch_size,
        )
        trainer.fit(model, datamodule)

        return trainer.callback_metrics["val_r2"].item()
