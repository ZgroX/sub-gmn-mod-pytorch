import math

import pytorch_lightning as pl
import torch
from torch import nn
from torchmetrics import Accuracy, F1Score, AUROC, PrecisionRecallCurve, AveragePrecision
import torch
import losses
import regularization
from subgmn import sub_GMN

lr = 0.01
weight_decay = 0.01


class Model(pl.LightningModule):
    full_state_update = False

    def __init__(self, GCN_in, GCN_out, NTN_k, GCN_k):
        super().__init__()
        self.model = sub_GMN(GCN_in, GCN_out, NTN_k, GCN_k)

        self.loss_function = losses.AsymmetricLoss()
        # self.loss_function = torch.nn.CrossEntropyLoss()
        self.xavier_init(self.model)
        # self.reg = regularization.Regularization(self.model, weight_decay=weight_decay, p=0)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        # return torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x_d, edge_index_d, x_q, edge_index_q, x_d_batch, x_q_batch, x_d_pos,
                x_q_pos):
        return self.model(x_d, edge_index_d, x_q, edge_index_q, x_d_batch, x_q_batch, x_d_pos,
                          x_q_pos)

    def training_step(self, batch, batch_idx):
        loss, acc, f1_s05, f1_s08, f1_s02, auc, y_min, y_max = self._shared_eval_step(batch, batch_idx)

        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_AUC", auc, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_05_score", f1_s05, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_08_score", f1_s08, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_f1_02_score", f1_s02, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_min", y_min, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_max", y_max, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    # TODO other steps, focusing now on training_step
    def validation_step(self, batch, batch_idx):
        loss, acc, f1_s05, f1_s08, f1_s02, auc, y_min, y_max = self._shared_eval_step(batch, batch_idx)

        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_AUC", auc, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_05_score", f1_s05, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_08_score", f1_s08, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_f1_02_score", f1_s02, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_min", y_min, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_max", y_max, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, acc, f1_s05, f1_s08, f1_s02, auc, y_min, y_max = self._shared_eval_step(batch, batch_idx)

        self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_AUC", auc, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_05_score", f1_s05, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_08_score", f1_s08, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_f1_02_score", f1_s02, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_min", y_min, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_max", y_max, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x_d, edge_index_d, x_d_batch, x_q, edge_index_q, x_q_batch, y, x_d_pos, x_q_pos = self._unpack_batch(batch)

        y_q_da_hat, y_da_q_hat = self.model(x_d, edge_index_d, x_q, edge_index_q, x_d_batch, x_q_batch, x_d_pos,
                                            x_q_pos)

        y = [torch.Tensor(y_t).to(self.device) for y_t in y]

        y_sum_0 = [yt.sum(dim=0) for yt in y]
        y_sum_1 = [yt.sum(dim=1) for yt in y]

        y_da_q_div = [torch.nan_to_num(yt / s, nan=1/yt.shape[1]) for yt, s in zip(y, y_sum_0)]
        y_q_da_div = [torch.nan_to_num(yt.T / s, nan=1/yt.shape[0]) for yt, s in zip(y, y_sum_1)]

        loss = self._loss_step(y_q_da_hat, y_q_da_div) + self._loss_step(y_da_q_hat, y_da_q_div)

        # loss = self._loss_step(y_q_da_hat, y_q_da) + self._loss_step(y_da_q_hat, y_da_q)
        # if not self.is_softmax:
        #     loss += self.reg(self.model)

        # -----------metrics--------------
        y_da_q = y
        y_q_da = [yt.T for yt in y]

        y_sum_0[y_sum_0 == 0] = 1
        y_sum_1[y_sum_1 == 0] = 1

        y_q_da_hat = [y_h * s for y_h, s in zip(y_q_da_hat, y_sum_1)]
        y_da_q_hat = [y_h * s for y_h, s in zip(y_da_q_hat, y_sum_0)]

        accuracy = Accuracy(task="binary").to(self.device)

        # [accuracy.update(y_h * s, y_t * s) for y_h, y_t, s in zip(y_q_da_hat, y_q_da, y_sum_1)]
        # [accuracy.update(y_h * s, y_t * s) for y_h, y_t, s in zip(y_da_q_hat, y_da_q, y_sum_0)]

        [accuracy.update(y_h, y_t) for y_h, y_t in zip(y_q_da_hat, y_q_da)]
        [accuracy.update(y_h, y_t) for y_h, y_t in zip(y_da_q_hat, y_da_q)]
        acc = accuracy.compute()

        f1 = F1Score(task="binary").to(self.device)

        [f1.update(y_h, y_t) for y_h, y_t in zip(y_q_da_hat, y_q_da)]
        [f1.update(y_h, y_t) for y_h, y_t in zip(y_da_q_hat, y_da_q)]
        f1_s05 = f1.compute()

        f1 = F1Score(task="binary", threshold=0.8).to(self.device)

        [f1.update(y_h, y_t) for y_h, y_t in zip(y_q_da_hat, y_q_da)]
        [f1.update(y_h, y_t) for y_h, y_t in zip(y_da_q_hat, y_da_q)]
        f1_s08 = f1.compute()

        f1 = F1Score(task="binary", threshold=0.2).to(self.device)

        [f1.update(y_h, y_t) for y_h, y_t in zip(y_q_da_hat, y_q_da)]
        [f1.update(y_h, y_t) for y_h, y_t in zip(y_da_q_hat, y_da_q)]
        f1_s02 = f1.compute()

        auc = AveragePrecision(task="binary")
        [auc.update(y_h, y_t.type(torch.int)) for y_h, y_t in zip(y_q_da_hat, y_q_da)]
        [auc.update(y_h, y_t.type(torch.int)) for y_h, y_t in zip(y_da_q_hat, y_da_q)]
        auc_s = auc.compute()

        return loss.to(self.device), acc.to(self.device), f1_s05.to(self.device), f1_s08.to(self.device), \
               f1_s02.to(self.device), auc_s.to(self.device), y_q_da_hat[0].min(), y_q_da_hat[0].max()

    def predict_step(self, batch, batch_idx, **kwargs):
        x, y = batch
        pred = self(x)
        return pred

    def _unpack_batch(self, batch):
        x_d = batch.x_d[0].type(torch.float32)
        edge_index_d = batch.edge_index_d[0].type(torch.int64)
        x_d_batch = batch.x_d_batch[0]
        x_d_pos = batch.pos_d[0]

        x_q = batch.x_q[0].type(torch.float32)
        edge_index_q = batch.edge_index_q[0].type(torch.int64)
        x_q_batch = batch.x_q_batch[0]
        x_q_pos = batch.pos_q[0]

        y = batch.truth_matrix

        return x_d, edge_index_d, x_d_batch, x_q, edge_index_q, x_q_batch, y, x_d_pos, x_q_pos

    def xavier_init(self, model):
        for name, param in model.named_parameters():
            if name.endswith(".bias"):
                param.data.fill_(0)
            else:
                bound = math.sqrt(6) / math.sqrt(param.shape[0] + param.shape[1])
                param.data.uniform_(-bound, bound)

    def _loss_step(self, y_hat, y):
        loss = self.loss_function(y_hat[0], y[0]).to(self.device)
        for y_h, y_t in zip(y_hat[1:], y[1:]):
            loss += self.loss_function(y_h, y_t).to(self.device)
        loss /= len(y)
        return loss
