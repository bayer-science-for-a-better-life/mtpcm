import torch
import numpy as np
from torch.utils.data import Dataset


class PCM_data_ST(Dataset):
    def __init__(self, data):
        self.cmp = data["cmp"].copy()
        self.prt = data["prt"].copy()
        self.asy = data["asy"].copy()
        self.pIC50 = data["pIC50"].copy()
        self.prefixes = data["prefixes"].copy()

    def normalise(self, params):

        self.prt = (self.prt - params["prt_mean"]) / params["prt_std"]
        self.asy = (self.asy - params["asy_mean"]) / params["asy_std"]
        self.pIC50 = (self.pIC50 - params["pIC50_mean"]) / params["pIC50_std"]

    def __len__(self):
        return len(self.cmp)

    def __getitem__(self, index: int):
        cmp = self.cmp[index]
        prt = self.prt[index]
        asy = self.asy[index]
        pIC50 = self.pIC50[index]
        prefixes = self.prefixes[index]

        return cmp, prt, asy, pIC50, prefixes


class PCM_data_MT(Dataset):
    def __init__(self, data):
        self.cmp = data["cmp"].copy()
        self.pIC50 = data["pIC50"].copy()
        self.prefixes = data["prefixes"].copy()
        self.taskind = data["taskind"].copy()

    def normalise(self, params):

        self.pIC50 = (self.pIC50 - params["pIC50_mean"]) / params["pIC50_std"]

    def __len__(self):
        return len(self.cmp)

    def __getitem__(self, index: int):
        cmp = self.cmp[index]
        pIC50 = self.pIC50[index]
        prefixes = self.prefixes[index]
        taskind = self.taskind[index]

        return cmp, pIC50, prefixes, taskind


class PCM_data_MT_withPRT(PCM_data_MT):
    def __init__(self, data):
        super().__init__(data)
        self.prt = data["prt"]

    def normalise(self, params):
        self.prt = (self.prt - params["prt_mean"]) / params["prt_std"]
        self.pIC50 = (self.pIC50 - params["pIC50_mean"]) / params["pIC50_std"]

    def __getitem__(self, index: int):
        cmp = self.cmp[index]
        prt = self.prt[index]
        pIC50 = self.pIC50[index]
        prefixes = self.prefixes[index]
        taskind = self.taskind[index]

        return cmp, prt, pIC50, prefixes, taskind
