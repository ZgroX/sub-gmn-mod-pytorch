import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import dgl
import networkx as nx
import dgl.function as fn
from dgl.data import MiniGCDataset
import dgl.function as fn
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dgl.nn.pytorch import SumPooling
import numpy as np
from dgl.data.utils import save_graphs, get_download_dir, load_graphs
# from dgl.subgraph import DGLSubGraph
from torch.utils.data import Dataset, DataLoader
# from dset import dgraph, collate
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import Linear
from dgl.nn.pytorch.conv import GraphConv
from torch.nn import Linear
from layers import three_gcn, att_layer, NTN
import torch_geometric.nn as g_nn
import torch_geometric.utils as g_utils


class sub_GMN(torch.nn.Module):
    def __init__(self, GCN_in_size, GCN_out_size, NTN_k, GCN_k):
        super(sub_GMN, self).__init__()
        self.GCN_in_size = GCN_in_size
        self.GCN_out_size = GCN_out_size  # D
        self.NTN_k = NTN_k
        self.GCN_k = GCN_k
        # layers
        self.GCN = three_gcn(in_size=self.GCN_in_size, out_size=self.GCN_out_size, number=self.GCN_k)
        self.NTN1 = NTN(D=self.GCN_out_size, k=self.NTN_k)
        self.NTN2 = NTN(D=self.GCN_out_size, k=self.NTN_k)
        self.NTN3 = NTN(D=self.GCN_out_size, k=self.NTN_k)

        self.Con1 = nn.Conv2d(self.NTN_k, 1, (1, 1))
        self.Con2 = nn.Conv2d(self.NTN_k, 1, (1, 1))
        self.Con3 = nn.Conv2d(self.NTN_k, 1, (1, 1))
        self.con_end = nn.Conv2d(self.NTN_k * GCN_k, 1, (1, 1))

    def forward(self, x_d, edge_index_d, x_q, edge_index_q, batch_index_d, batch_index_q, x_d_pos,
                 x_q_pos):  # b_same bx5x8

        da_embeddings, q_embeddings = self._embedding_step(x_d, edge_index_d, x_q, edge_index_q, batch_index_d,
                                                         batch_index_q, x_d_pos, x_q_pos)

        del x_d, edge_index_d, x_q, edge_index_q, batch_index_d, batch_index_q, x_d_pos, x_q_pos

        N_q_da = []
        N_da_q = []

        for da, q in zip(da_embeddings, q_embeddings):
            N_q_da.append(self._attention_step(q, da))
            N_da_q.append(self._attention_step(da, q))
        #
        # N1_q_da = self._attention_step(q1, da1)
        # N1_da_q = self._attention_step(da1, q1)
        # del da1, q1
        #
        # N2_q_da = self._attention_step(q2, da2)
        # N2_da_q = self._attention_step(da2, q2)
        # del da2, q2
        #
        # N3_q_da = self._attention_step(q3, da3)
        # N3_da_q = self._attention_step(da3, q3)
        # del da3, q3

        end_q_da = self._conv_step(N_q_da)
        end_da_q = self._conv_step(N_da_q)


        return end_q_da, end_da_q

    def _embedding_step(self, x_d, edge_index_d, x_q, edge_index_q, batch_index_d, batch_index_q, x_d_pos, x_q_pos):
        x_d_data = torch_geometric.data.Data(x=x_d, edge_index=edge_index_d, pos=x_d_pos)
        x_q_data = torch_geometric.data.Data(x=x_q, edge_index=edge_index_q, pos=x_q_pos)
        x_d_data = torch_geometric.transforms.Cartesian()(x_d_data)
        x_q_data = torch_geometric.transforms.Cartesian()(x_q_data)

        da_embeddings = self.GCN(x_d_data.x, x_d_data.edge_index, x_d_data.edge_attr)
        q_embeddings = self.GCN(x_q_data.x, x_q_data.edge_index, x_q_data.edge_attr)

        for i,da in enumerate(da_embeddings):
            da_embeddings[i] = g_utils.unbatch(da, batch_index_d)

        for i,q in enumerate(q_embeddings):
            q_embeddings[i] = g_utils.unbatch(q, batch_index_q)

        # da1 = g_utils.unbatch(da1, batch_index_d)
        # da2 = g_utils.unbatch(da2, batch_index_d)
        # da3 = g_utils.unbatch(da3, batch_index_d)

        # q1 = g_utils.unbatch(q1, batch_index_q)
        # q2 = g_utils.unbatch(q2, batch_index_q)
        # q3 = g_utils.unbatch(q3, batch_index_q)

        return da_embeddings, q_embeddings

    def _attention_step(self, q, da):
        att1 = att_layer(batch_q_em=q, batch_da_em=da)  # att bx1x5x18
        N1_16 = self.NTN1(q, da)  # N1_16 bxkx5x18
        return [n * a for n, a in zip(N1_16, att1)]

    def _conv_step(self, Ns):
        end = [torch.cat([*n], dim=0) for n in
               zip(*Ns)]  # end bx(3k+6)x5x18
        end = [self.con_end(e) for e in end]  # end bx1x5x18
        end = [e.squeeze() for e in end]  # end bx5x18
        end = [torch.softmax(e, dim=0) for e in end]

        return end
