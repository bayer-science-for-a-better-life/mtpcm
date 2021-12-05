# pylint: disable=no-member

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import math


class CensoredMSELoss_less(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return (y - y_hat) * torch.nn.ReLU()(y - y_hat)


class CensoredMSELoss_more(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, y):
        return (y_hat - y) * torch.nn.ReLU()(y_hat - y)


class CensoredMSELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lossfn_less = CensoredMSELoss_less()
        self.lossfn_more = CensoredMSELoss_more()

    def forward(self, yhat, y, prefixes):
        loss_pIC50_less = self.lossfn_less(yhat, y)
        loss_pIC50_more = self.lossfn_more(yhat, y)
        coeff_less = -torch.nn.ReLU()(-prefixes) + 1
        coeff_more = -torch.nn.ReLU()(prefixes) + 1
        return coeff_less * loss_pIC50_less + coeff_more * loss_pIC50_more


class MCDropoutCallback(Callback):
    def on_test_epoch_start(self, trainer, pl_module):
        print("MC callback")
        for m in pl_module.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()


class ValidationMCDropoutCallback(Callback):
    def on_validation_epoch_start(self, trainer, pl_module):
        print("MC callback")
        for m in pl_module.modules():
            if m.__class__.__name__.startswith("Dropout"):
                m.train()


class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_epoch_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


class SparseLinear(torch.nn.Module):
    """
    Linear layer with one-hot-encoding input tensor, containing index of non-zero element, and dense output.
        in_features    size of input
        out_features   size of output
        bias           whether to add bias
    """

    def __init__(self, in_features, out_features, bias=True):
        super(SparseLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn(
            in_features, out_features) / math.sqrt(out_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        out = torch.index_select(self.weight, 0, input.long())
        if self.bias is not None:
            return out + self.bias
        return out

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.weight.shape[0], self.weight.shape[1], self.bias is not None
        )
