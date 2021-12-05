# pylint: disable=no-member

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pytorch_lightning as pl
from PCM.utils import CensoredMSELoss
from scipy import stats
from sklearn.metrics import r2_score


class PCM_MT(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters()

        self.num_tasks = hparams["num_tasks"]
        self.activation = nn.ELU()

        self.lossfn = CensoredMSELoss()

        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        self.cmp_bottleneck = nn.Linear(512, hparams["cmp_size"])
        self.dropout_0 = nn.Dropout(hparams["dropout"])
        self.lr = hparams["lr"]

        input_dim = hparams["cmp_size"]
        for i in range(hparams["num_layers"]):
            output_dim = hparams["layers"][i]
            self.layers.append(nn.Linear(input_dim, output_dim))
            self.dropouts.append(nn.Dropout(hparams["dropout"]))
            input_dim = output_dim

        self.last_layer = nn.Linear(output_dim, self.num_tasks)

        self.seed0 = np.random.randint(1e10)
        self.num_test_trials = 5

    def forward(self, cmp, seed=None, last_layer=True):
        if seed is not None:
            torch.manual_seed(seed)
        cmp_bottle = self.dropout_0(self.activation(self.cmp_bottleneck(cmp)))

        x = cmp_bottle

        for layer, dropout in zip(self.layers, self.dropouts):
            x = dropout(self.activation(layer(x)))

        if last_layer:
            x = self.last_layer(x)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer

    def loss(self, yhat, y, prefixes):

        return torch.mean(self.lossfn(yhat, y, prefixes))

    def training_step(self, train_batch, batch_idx):

        x_cmp, y, y_cen, y_ind = train_batch
        outputs = self.forward(x_cmp.float())
        y_hat = outputs[range(len(y_ind)), y_ind]

        loss = self.loss(y_hat.view(-1), y, y_cen)
        self.log("train_loss", loss)

        return {"loss": loss}

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_train_loss", avg_loss)
        return

    def validation_step(self, val_batch, batch_idx):

        x_cmp, y, _, y_ind = val_batch
        outputs = self.forward(x_cmp.float())
        y_hat = outputs[range(len(y_ind)), y_ind]

        return {"val_yhat": y_hat, "val_y": y}

    def validation_epoch_end(self, outputs):

        yhat = np.hstack([x["val_yhat"].cpu().numpy() for x in outputs])
        y = np.hstack([x["val_y"].cpu().numpy() for x in outputs])

        val_sp = stats.spearmanr(yhat, y)[0]
        val_r2 = r2_score(y, yhat)

        self.log("val_spearmanr", val_sp)
        self.log("val_r2", val_r2)
        return

    def test_step(self, test_batch, batch_idx):

        x_cmp, y, y_cen, y_ind = test_batch
        pred = np.zeros((len(y), self.num_test_trials))
        for i in range(self.num_test_trials):
            outputs = self.forward(
                x_cmp.float(), seed=self.seed0 + self.current_epoch + i
            )
            y_hat = outputs[range(len(y_ind)), y_ind]
            pred[:, i] = y_hat.detach().cpu().numpy().squeeze()
        y_hat = np.mean(pred, axis=1).squeeze()
        y_hat_std = np.std(pred, axis=1).squeeze()

        return {
            "predictions": y_hat,
            "predictions_std": y_hat_std,
            "test_ycen": y_cen,
            "test_y": y,
            "test_yind": y_ind,
        }

    def test_epoch_end(self, outputs):

        yhat = np.hstack([x["predictions"] for x in outputs])
        yhat_std = np.hstack([x["predictions_std"] for x in outputs])
        y = np.hstack([x["test_y"].cpu().numpy() for x in outputs])
        y_cen = np.hstack([x["test_ycen"].cpu().numpy() for x in outputs])
        y_ind = np.hstack([x["test_yind"].cpu().numpy() for x in outputs])

        spearmanr = torch.Tensor([stats.spearmanr(yhat[y_cen == 0], y[y_cen == 0])[0]])
        print("Test Spearman correlation is %.3lf" % (spearmanr))

        self.test_predictions = yhat
        self.test_predictions_std = yhat_std
        self.test_predictions_ind = y_ind

        return


class PCM_MT_withPRT(PCM_MT):
    def __init__(self, hparams):
        super().__init__(hparams)

        self.prt_bottleneck = nn.Linear(256, hparams["prt_size"])

        self.layers = nn.ModuleList()
        input_dim = hparams["prt_size"] + hparams["cmp_size"]
        for i in range(hparams["num_layers"]):
            output_dim = hparams["layers"][i]
            self.layers.append(nn.Linear(input_dim, output_dim))
            input_dim = output_dim

        self.last_layer = nn.Linear(output_dim, self.num_tasks)

    def forward(self, cmp, prt, seed=None, last_layer=True):
        if seed is not None:
            torch.manual_seed(seed)
        cmp_bottle = self.dropout_0(self.activation(self.cmp_bottleneck(cmp)))
        prt_bottle = self.dropout_0(self.activation(self.prt_bottleneck(prt)))

        x = torch.cat((cmp_bottle, prt_bottle), dim=1)

        for layer, dropout in zip(self.layers, self.dropouts):
            x = dropout(self.activation(layer(x)))

        if last_layer:
            x = self.last_layer(x)
        return x

    def training_step(self, train_batch, batch_idx):

        x_cmp, x_prt, y, y_cen, y_ind = train_batch
        outputs = self.forward(x_cmp.float(), x_prt.float())
        y_hat = outputs[range(len(y_ind)), y_ind]

        loss = self.loss(y_hat.view(-1), y, y_cen)
        self.log("train_loss", loss)

        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):

        x_cmp, x_prt, y, _, y_ind = val_batch
        outputs = self.forward(x_cmp.float(), x_prt.float())
        y_hat = outputs[range(len(y_ind)), y_ind]

        return {"val_yhat": y_hat, "val_y": y}

    def test_step(self, test_batch, batch_idx):

        x_cmp, x_prt, y, y_cen, y_ind = test_batch
        pred = np.zeros((len(y), self.num_test_trials))
        for i in range(self.num_test_trials):
            outputs = self.forward(
                x_cmp.float(), x_prt.float(), seed=self.seed0 + self.current_epoch + i
            )
            y_hat = outputs[range(len(y_ind)), y_ind]
            pred[:, i] = y_hat.detach().cpu().numpy().squeeze()
        y_hat = np.mean(pred, axis=1).squeeze()
        y_hat_std = np.std(pred, axis=1).squeeze()

        return {
            "predictions": y_hat,
            "predictions_std": y_hat_std,
            "test_ycen": y_cen,
            "test_y": y,
            "test_yind": y_ind,
        }
