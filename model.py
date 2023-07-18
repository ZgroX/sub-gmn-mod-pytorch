import math

import pytorch_lightning as pl
import torch

from torch import nn
from torchmetrics import Accuracy, F1Score

import losses
import regularization
from subgmn import sub_GMN

lr = 0.001
weight_decay = 0.01


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Expand(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.expand(self.shape)


class Model(pl.LightningModule):
    full_state_update = False

    def __init__(self, GCN_in, GCN_out, NTN_k, is_softmax):
        super().__init__()
        self.is_softmax = is_softmax
        self.model = sub_GMN(GCN_in, GCN_out, NTN_k, is_softmax)

        if is_softmax:
            self.loss_function = torch.nn.MSELoss()
        else:
            self.loss_function = losses.AsymmetricLoss()
            self.xavier_init(self.model)
            self.reg = regularization.Regularization(self.model, weight_decay=weight_decay, p=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x, y):
        return self.model(x, y)

    def training_step(self, batch, batch_idx):
        x_d, edge_index_d, x_d_batch, x_q, edge_index_q, x_q_batch, y = self._unpack_batch(batch)

        y_hat = self.model(x_d, edge_index_d, x_q, edge_index_q, x_d_batch, x_q_batch)
        y = [torch.Tensor(y_t).to(self.device) for y_t in y]

        if self.is_softmax:
            yy = [[torch.masked_select(y_h.T, y_t.type(torch.bool)).to(self.device), torch.masked_select(y_t, y_t.type(torch.bool)).to(self.device)] for y_h, y_t in zip(y_hat, y)]
        else:
            yy = [[y_h.T, y_t] for y_h, y_t in zip(y_hat, y)]

        y_hat = [x for x, _ in yy]
        y = [x for _, x in yy]

        loss = self.loss_function(y_hat[0], y[0]).to(self.device)
        for y_h, y_t in zip(y_hat[1:], y[1:]):
            loss += self.loss_function(y_h, y_t).to(self.device)
        loss /= len(y)
        if not self.is_softmax:
            loss += self.reg(self.model)
        accuracy = Accuracy(task="binary").to(self.device)
        [accuracy.update(y_h, y_t) for y_h, y_t in zip(y_hat, y)]
        acc = accuracy.compute()
        f1 = F1Score(task="binary").to(self.device)
        [f1.update(y_h, y_t) for y_h, y_t in zip(y_hat, y)]
        f1_s = f1.compute()

        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_score", f1_s, on_epoch=True, prog_bar=True, logger=True)
        self.log("loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss.to(self.device)

    # TODO other steps, focusing now on training_step
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_test_step(batch, batch_idx)  # , prec, rec, f1
        self.log("test_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("test_accuracy", acc, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_precision", prec, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_recall", rec, on_epoch=True, prog_bar=True, logger=True)
        # self.log("test_f1_score", f1, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x_d, edge_index_d, x_d_batch, x_q, edge_index_q, x_q_batch, y = self._unpack_batch(batch)

        y_hat = self.model(x_d, edge_index_d, x_q, edge_index_q, x_d_batch, x_q_batch)
        loss = (
            torch.Tensor([nn.MSELoss()(y_h, torch.Tensor(y_t.T).to(self.device)) for y_h, y_t in zip(y_hat, y)])).to(
            self.device).mean()
        accuracy = Accuracy(task="binary", num_classes=4).to(self.device)
        [accuracy.update(y_h, torch.Tensor(y_t.T).to(self.device)) for y_h, y_t in zip(y_hat, y)]
        acc = accuracy.compute()
        return loss.to(self.device), acc.to(self.device)

    def _shared_test_step(self, batch, batch_idx):
        x_d, edge_index_d, x_d_batch, x_q, edge_index_q, x_q_batch, y = self._unpack_batch(batch)

        y_hat = self.model(x_d, edge_index_d, x_q, edge_index_q, x_d_batch, x_q_batch)
        loss = (
            torch.Tensor([nn.MSELoss()(y_h, torch.Tensor(y_t.T).to(self.device)) for y_h, y_t in zip(y_hat, y)])).to(
            self.device).mean()
        accuracy = Accuracy(task="binary", num_classes=4).to(self.device)
        [accuracy.update(y_h, torch.Tensor(y_t.T).to(self.device)) for y_h, y_t in zip(y_hat, y)]
        acc = accuracy.compute()
        # acc = accuracy(y_hat, y, task="multiclass", num_classes=self.n_classes, average='weighted')
        # prec = precision(y_hat, y, task="multiclass", num_classes=self.n_classes, average='weighted')
        # rec = recall(y_hat, y, task="multiclass", num_classes=self.n_classes, average='weighted')
        # f1 = f1_score(y_hat, y, task="multiclass", num_classes=self.n_classes, average='weighted')
        return loss.to(self.device), acc.to(self.device)  # , prec, rec, f1

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        pred = self(x)
        return pred

    def deconv_step(self, batch):
        x, y = batch
        deconv = self.deconv(x)
        return deconv

    def _unpack_batch(self, batch):
        x_d = batch.x_d[0].type(torch.float32)
        edge_index_d = batch.edge_index_d[0].type(torch.int64)
        x_d_batch = batch.x_d_batch[0]

        x_q = batch.x_q[0].type(torch.float32)
        edge_index_q = batch.edge_index_q[0].type(torch.int64)
        x_q_batch = batch.x_q_batch[0]

        y = batch.truth_matrix

        return x_d, edge_index_d, x_d_batch, x_q, edge_index_q, x_q_batch, y

    def xavier_init(self, model):
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            else:
                bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                param.data.uniform_(-bound, bound)
