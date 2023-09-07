import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from torchmetrics import AveragePrecision
from tqdm import tqdm

from traffic_signs_datamodule import TrafficSignDataModule
from model import Model
from datetime import datetime
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from playground import draw_graphs_from_pair

GCN_IN = 3
GCN_OUT = 128
NTN_k = 32
GCN_k = 7

torch.set_float32_matmul_precision('medium')


def evaluate():
    size = (1300, 1300)
    dm = TrafficSignDataModule(batch_size=1, crop_size=size, data_graph_seg_size=400)
    dm.prepare_data()
    dm.setup()

    model = Model.load_from_checkpoint(GCN_in=GCN_IN, GCN_out=GCN_OUT, NTN_k=NTN_k, GCN_k=GCN_k,  checkpoint_path="lightning_logs/Sub-GMN/TrafficSign/checkpoints/2023-09-03_13_39_11.978569GCN_OUT=128--NTN_k=32-epoch=499--val_AUC=0.40.ckpt")

    test_data = dm.test_dataloader()
    #pair = list(test_data)[0]

    results = []

    for pair in tqdm(list(test_data)[:2]):
        x_d, edge_index_d, x_d_batch, x_q, edge_index_q, x_q_batch, y, x_d_pos, x_q_pos = model._unpack_batch(pair)
        y_q_da_hat, y_da_q_hat = model(x_d, edge_index_d, x_q, edge_index_q, x_d_batch, x_q_batch, x_d_pos,
                    x_q_pos)

        y = torch.Tensor(y[0])

        y_sum_0 = y.sum(dim=0)
        y_sum_1 = y.sum(dim=1)

        y_sum_0[y_sum_0 == 0] = 1
        y_sum_1[y_sum_1 == 0] = 1

        y_q_da_hat = y_q_da_hat[0] * y_sum_1
        y_da_q_hat = y_da_q_hat[0] * y_sum_0

        y_q_da_hat = y_q_da_hat.T

        tm = torch.maximum(y_q_da_hat, y_da_q_hat)

        auc = AveragePrecision(task="binary")
        [auc.update(y_h, y_t.type(torch.int)) for y_h, y_t in zip(tm, y)]
        auc_max = auc.compute()
        results.append([pair, auc_max.item()])

    results_np = np.array(list(zip(*results))[1])

    results_max = results_np.argmax()
    results_min = results_np.argmin()
    results_avg = np.argsort(results_np)[len(results_np)//2]

    results = [results[results_max], results[results_min], results[results_avg]]
    strings = ["max", "min", "avg"]

    for result, strin in zip(results, strings):
        pair = result[0]
        auc_score = result[1]

        x_d, edge_index_d, x_d_batch, x_q, edge_index_q, x_q_batch, y, x_d_pos, x_q_pos = model._unpack_batch(pair)
        y_q_da_hat, y_da_q_hat = model(x_d, edge_index_d, x_q, edge_index_q, x_d_batch, x_q_batch, x_d_pos,
                                       x_q_pos)

        y = torch.Tensor(y[0])

        y_sum_0 = y.sum(dim=0)
        y_sum_1 = y.sum(dim=1)

        y_sum_0[y_sum_0 == 0] = 1
        y_sum_1[y_sum_1 == 0] = 1

        y_q_da_hat = y_q_da_hat[0] * y_sum_1
        y_da_q_hat = y_da_q_hat[0] * y_sum_0

        y_q_da_hat = y_q_da_hat.T.round()
        y_da_q_hat = y_da_q_hat.round()

        tm = torch.maximum(y_q_da_hat, y_da_q_hat)

        tm = tm.detach().numpy()



        draw_graphs_from_pair(pair, pair.truth_matrix, f"original_eval_graphs_{strin}_{auc_score=}", size=size)
        draw_graphs_from_pair(pair, [tm], f"model_eval_graphs_max_{strin}_{auc_score=}", size=size)
        draw_graphs_from_pair(pair, [y_q_da_hat], f"model_eval_graphs_y_q_da_hat_{strin}_{auc_score=}", size=size)
        draw_graphs_from_pair(pair, [y_da_q_hat], f"model_eval_graphs_y_da_q_hat_{strin}_{auc_score=}", size=size)


if __name__ == '__main__':
    evaluate()
