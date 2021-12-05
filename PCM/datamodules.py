import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from PCM.datasets import PCM_data_ST, PCM_data_MT, PCM_data_MT_withPRT
from PCM.dataloaders import (
    GPU_DataLoader_ST,
    GPU_DataLoader_MT,
    GPU_DataLoader_MT_withPRT,
)

# pylint: disable=no-member


class PCM_DataModule_ST(pl.LightningDataModule):
    def __init__(
        self,
        data_train,
        data_val,
        data_test=None,
        censored=True,
        cmp_noise=0.0,
        prt_noise=0.0,
        batch_size=512,
        test_batch_size=None,
        norm_asy=True,
    ):
        super().__init__()
        self.cmp_noise = cmp_noise
        self.prt_noise = prt_noise
        self.data_train = PCM_data_ST(data_train)
        self.data_val = PCM_data_ST(data_val)
        if data_test is None:
            self.data_test = None
        else:
            self.data_test = PCM_data_ST(data_test)
        self.censored = censored
        self.batch_size = batch_size
        if test_batch_size is not None:
            self.test_batch_size = test_batch_size
        else:
            self.test_batch_size = batch_size

        self.norm_params = {
            "prt_mean": self.data_train.prt.mean(axis=0),
            "prt_std": self.data_train.prt.std(axis=0),
            "asy_mean": self.data_train.asy.mean(axis=0),
            "asy_std": self.data_train.asy.std(axis=0),
            "pIC50_mean": np.mean(self.data_train.pIC50),
            "pIC50_std": np.std(self.data_train.pIC50),
        }

        if not norm_asy:
            self.norm_params["asy_mean"] = np.zeros(self.norm_params["asy_mean"].shape)
            self.norm_params["asy_std"] = np.ones(self.norm_params["asy_std"].shape)

    def prepare_data(self):

        self.data_train.normalise(self.norm_params)
        self.data_val.normalise(self.norm_params)
        if self.data_test is not None:
            self.data_test.normalise(self.norm_params)

        if not self.censored:
            self.data_train.prefixes = np.zeros(self.data_train.prefixes.size)

        return

    def train_dataloader(self):
        return GPU_DataLoader_ST(
            self.data_train.cmp,
            self.data_train.prt,
            self.data_train.asy,
            self.data_train.pIC50,
            self.data_train.prefixes,
            self.cmp_noise,
            self.prt_noise,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
        )


class PCM_DataModule_MT(pl.LightningDataModule):
    def __init__(
        self,
        num_tasks,
        data_train,
        data_val,
        data_test=None,
        censored=True,
        cmp_noise=0.0,
        batch_size=512,
        test_batch_size=None,
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.cmp_noise = cmp_noise
        self.data_train = PCM_data_MT(data_train)
        self.data_val = PCM_data_MT(data_val)
        if data_test is None:
            self.data_test = None
        else:
            self.data_test = PCM_data_MT(data_test)
        self.censored = censored
        self.batch_size = batch_size
        if test_batch_size is not None:
            self.test_batch_size = test_batch_size
        else:
            self.test_batch_size = batch_size

        self.norm_params = {
            "pIC50_mean": np.mean(self.data_train.pIC50),
            "pIC50_std": np.std(self.data_train.pIC50),
        }

    def prepare_data(self):

        self.data_train.normalise(self.norm_params)
        self.data_val.normalise(self.norm_params)
        if self.data_test is not None:
            self.data_test.normalise(self.norm_params)

        if not self.censored:
            self.data_train.prefixes = torch.zeros(self.data_train.prefixes.size)

        return

    def train_dataloader(self):
        return GPU_DataLoader_MT(
            self.data_train.cmp,
            self.data_train.pIC50,
            self.data_train.prefixes,
            self.data_train.taskind,
            self.cmp_noise,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val, batch_size=self.batch_size, shuffle=False, drop_last=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.test_batch_size,
            shuffle=False,
            drop_last=False,
        )


class PCM_DataModule_MT_withPRT(PCM_DataModule_MT):
    def __init__(
        self,
        num_tasks,
        data_train,
        data_val,
        data_test=None,
        censored=True,
        cmp_noise=0.0,
        prt_noise=0.0,
        batch_size=512,
        test_batch_size=None,
    ):
        super().__init__(
            num_tasks,
            data_train,
            data_val,
            data_test,
            censored,
            cmp_noise,
            batch_size,
            test_batch_size,
        )
        self.prt_noise = prt_noise
        self.data_train = PCM_data_MT_withPRT(data_train)
        self.data_val = PCM_data_MT_withPRT(data_val)
        if data_test is not None:
            self.data_test = PCM_data_MT_withPRT(data_test)
        else:
            data_test = None

        self.norm_params = {
            "prt_mean": self.data_train.prt.mean(axis=0),
            "prt_std": self.data_train.prt.std(axis=0),
            "pIC50_mean": np.mean(self.data_train.pIC50),
            "pIC50_std": np.std(self.data_train.pIC50),
        }

    def train_dataloader(self):
        return GPU_DataLoader_MT_withPRT(
            self.data_train.cmp,
            self.data_train.prt,
            self.data_train.pIC50,
            self.data_train.prefixes,
            self.data_train.taskind,
            self.cmp_noise,
            self.prt_noise,
            batch_size=self.batch_size,
            shuffle=True,
        )
