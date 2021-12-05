# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from PCM.utils import CensoredMSELoss, SparseLinear
from scipy import stats
from sklearn.metrics import r2_score
from torch.nn import MSELoss


class PCM(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()

        self.activation = nn.ELU()

        self.lossfn = CensoredMSELoss()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.prt_bottleneck = nn.Linear(256, hparams["prt_size"])
        self.cmp_bottleneck = nn.Linear(512, hparams["cmp_size"])
        self.asy_bottleneck = nn.Linear(2, 8)
        self.dropout_0 = nn.Dropout(hparams["dropout"])
        self.lr = hparams["lr"]

        input_dim = hparams["prt_size"] + hparams["cmp_size"] + 8
        for i in range(hparams["num_layers"]):
            output_dim = hparams["layers"][i]
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(hparams["dropout"]))
            input_dim = output_dim

        self.last_layer = nn.Linear(output_dim, 1)

        self.seed0 = np.random.randint(1e+10)
        self.num_test_trials = 5

    def forward(self, cmp, prt, asy, seed=None, last_layer=True):
        if seed is not None:
            torch.manual_seed(seed)
        cmp_bottle = self.dropout_0(self.activation(self.cmp_bottleneck(cmp)))
        prt_bottle = self.dropout_0(self.activation(self.prt_bottleneck(prt)))
        asy_bottle = self.dropout_0(self.activation(self.asy_bottleneck(asy)))

        x = torch.cat((cmp_bottle, prt_bottle, asy_bottle), dim=1)

        for layer, dropout in zip(self.layers, self.dropouts):
            x = dropout(self.activation(layer(x)))
        if last_layer:
            x = self.last_layer(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, yhat, y, prefixes):

        # torch.mean(self.lossfn(yhat, y, prefixes))
        return torch.mean(self.lossfn(yhat, y, prefixes))

    def training_step(self, train_batch, batch_idx):

        x_cmp, x_prt, x_asy, y, y_cen = train_batch
        y_hat = self.forward(x_cmp.float(), x_prt.float(), x_asy.float())

        loss = self.loss(y_hat.view(-1), y, y_cen)

        self.log('train_loss', loss)

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss)
        return

    def validation_step(self, val_batch, batch_idx):

        x_cmp, x_prt, x_asy, y, _ = val_batch

        y_hat = self.forward(x_cmp.float(), x_prt.float(), x_asy.float())

        return {"val_yhat": y_hat, "val_y": y}

    def validation_epoch_end(self, outputs):

        yhat = np.hstack([x["val_yhat"].cpu().numpy().squeeze()
                          for x in outputs])
        y = np.hstack([x["val_y"].cpu().numpy().squeeze() for x in outputs])

        val_sp = stats.spearmanr(yhat, y)[0]
        val_r2 = r2_score(y, yhat)

        self.log("val_r2", val_r2)
        self.log("val_spearmanr", val_sp)
        return

    def test_step(self, test_batch, batch_idx):

        x_cmp, x_prt, x_asy, y, y_cen = test_batch
        pred = np.zeros((len(y), self.num_test_trials))
        for i in range(self.num_test_trials):
            outputs = self.forward(x_cmp.float(),
                                   x_prt.float(),
                                   x_asy.float(),
                                   seed=self.seed0 + self.current_epoch + i)
            pred[:, i] = outputs.detach().cpu().numpy().squeeze()

        y_hat = np.mean(pred, axis=1).squeeze()
        y_hat_std = np.std(pred, axis=1).squeeze()

        return {
            "predictions": y_hat,
            "predictions_std": y_hat_std,
            "test_ycen": y_cen,
            "test_y": y,
        }

    def test_epoch_end(self, outputs):

        yhat = np.hstack([x["predictions"] for x in outputs])
        yhat_std = np.hstack([x["predictions_std"] for x in outputs])
        y = np.hstack([x["test_y"].cpu().numpy() for x in outputs])
        y_cen = np.hstack([x["test_ycen"].cpu().numpy() for x in outputs])

        spearmanr = stats.spearmanr(yhat[y_cen == 0], y[y_cen == 0])[0]
        print("Test Spearman correlation is %.3lf" % (spearmanr))

        self.test_predictions = yhat
        self.test_predictions_std = yhat_std

        return


class PCM_ext(PCM):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.asy_bottleneck = SparseLinear(hparams['num_tasks'], 8)
