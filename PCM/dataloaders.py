import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# pylint: disable=no-member


class GPU_DataLoader_ST(object):
    def __init__(
        self,
        cmp,
        prt,
        asy,
        pIC50,
        prefixes,
        cmp_noise=0.0,
        prt_noise=0.0,
        batch_size=512,
        shuffle=False,
    ):

        self._n_examples = len(cmp)
        self.cmp_noise = cmp_noise
        self.prt_noise = prt_noise

        self.cmp = torch.tensor(cmp).float().cuda()
        self.prt = torch.tensor(prt).float().cuda()
        self.asy = torch.tensor(asy).float().cuda()
        self.pIC50 = torch.tensor(pIC50).float().cuda()
        self.prefixes = torch.tensor(prefixes).float().cuda()

        self.batch_index = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self._update_permutation()
        else:
            self.permutation = torch.arange(self._n_examples).cuda()

    def __iter__(self):
        return self

    def _update_permutation(self):

        self.permutation = torch.randperm(self._n_examples).cuda()

    def _get_batch(self):
        example_indices = self.permutation.narrow(
            dim=0, start=self.batch_index, length=self.batch_size
        )
        cmp = torch.index_select(self.cmp, dim=0, index=example_indices)
        prt = torch.index_select(self.prt, dim=0, index=example_indices)
        asy = torch.index_select(self.asy, dim=0, index=example_indices)
        pIC50 = torch.index_select(self.pIC50, dim=0, index=example_indices)
        prefixes = torch.index_select(self.prefixes, dim=0, index=example_indices)

        cmp = torch.tanh(
            torch.atanh(cmp) + (torch.randn_like(cmp).cuda() * self.cmp_noise)
        )
        prt = prt + (torch.randn_like(prt).cuda() * self.prt_noise)

        return cmp, prt, asy, pIC50, prefixes

    def __next__(self):

        if self.batch_index + self.batch_size > self._n_examples:
            self.batch_index = 0
            if self.shuffle:
                self._update_permutation()
            raise StopIteration
        else:
            batch = self._get_batch()
            self.batch_index += self.batch_size
            return batch


class GPU_DataLoader_MT(object):
    def __init__(
        self,
        cmp,
        pIC50,
        prefixes,
        taskind,
        cmp_noise=0.0,
        batch_size=512,
        shuffle=False,
    ):

        self._n_examples = len(cmp)
        self.cmp_noise = cmp_noise

        self.cmp = torch.tensor(cmp).float().cuda()
        self.pIC50 = torch.tensor(pIC50).float().cuda()
        self.prefixes = torch.tensor(prefixes).float().cuda()
        self.taskind = torch.tensor(taskind).long().cuda()

        self.batch_index = 0
        self.batch_size = batch_size
        self.shuffle = shuffle
        if self.shuffle:
            self._update_permutation()
        else:
            self.permutation = torch.arange(self._n_examples).cuda()

    def __iter__(self):
        return self

    def _update_permutation(self):

        self.permutation = torch.randperm(self._n_examples).cuda()

    def _get_batch(self):
        example_indices = self.permutation.narrow(
            dim=0, start=self.batch_index, length=self.batch_size
        )
        cmp = torch.index_select(self.cmp, dim=0, index=example_indices)
        pIC50 = torch.index_select(self.pIC50, dim=0, index=example_indices)
        prefixes = torch.index_select(self.prefixes, dim=0, index=example_indices)
        taskind = torch.index_select(self.taskind, dim=0, index=example_indices)

        cmp = torch.tanh(
            torch.atanh(cmp) + (torch.randn_like(cmp).cuda() * self.cmp_noise)
        )

        return cmp, pIC50, prefixes, taskind

    def __next__(self):

        if self.batch_index + self.batch_size > self._n_examples:
            self.batch_index = 0
            if self.shuffle:
                self._update_permutation()
            raise StopIteration
        else:
            batch = self._get_batch()
            self.batch_index += self.batch_size
            return batch


class GPU_DataLoader_MT_withPRT(GPU_DataLoader_MT):
    def __init__(
        self,
        cmp,
        prt,
        pIC50,
        prefixes,
        task_ind,
        cmp_noise=0.0,
        prt_noise=0.0,
        batch_size=512,
        shuffle=False,
    ):

        super().__init__(cmp, pIC50, prefixes, task_ind, cmp_noise, batch_size, shuffle)
        self.prt_noise = prt_noise
        self.prt = torch.tensor(prt).float().cuda()

    def _get_batch(self):
        example_indices = self.permutation.narrow(
            dim=0, start=self.batch_index, length=self.batch_size
        )
        cmp = torch.index_select(self.cmp, dim=0, index=example_indices)
        prt = torch.index_select(self.prt, dim=0, index=example_indices)
        pIC50 = torch.index_select(self.pIC50, dim=0, index=example_indices)
        prefixes = torch.index_select(self.prefixes, dim=0, index=example_indices)
        taskind = torch.index_select(self.taskind, dim=0, index=example_indices)

        cmp = torch.tanh(
            torch.atanh(cmp) + (torch.randn_like(cmp).cuda() * self.cmp_noise)
        )
        prt = prt + (torch.randn_like(prt).cuda() * self.prt_noise)

        return cmp, prt, pIC50, prefixes, taskind