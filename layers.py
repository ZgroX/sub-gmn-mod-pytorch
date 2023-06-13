
import torch

import torch_geometric.nn as g_nn


class three_gcn(torch.nn.Module):
    def __init__(self, in_size, out_size):
        super(three_gcn, self).__init__()
        self.out_size = out_size
        self.gcn1 = g_nn.GraphConv(in_channels=in_size, out_channels=self.out_size)
        self.gcn2 = g_nn.GraphConv(in_channels=self.out_size, out_channels=self.out_size)
        self.gcn3 = g_nn.GraphConv(in_channels=self.out_size, out_channels=self.out_size)

    def forward(self, graph, edge_index):
        y1 = self.gcn1(x=graph, edge_index=edge_index)
        y1 = torch.nn.functional.elu(y1)
        y2 = self.gcn2(x=y1, edge_index=edge_index)
        y2 = torch.nn.functional.elu(y2)
        y3 = self.gcn3(x=y2, edge_index=edge_index)
        y3 = torch.nn.functional.sigmoid(y3)
        return y1, y2, y3


def att_layer(batch_q_em, batch_da_em):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
    #
    # D = batch_q_em.size()[2]
    # T_batch_da_em = torch.transpose(batch_da_em, 1, 2)
    # att = torch.matmul(batch_q_em, T_batch_da_em)
    # att = att / (D ** 0.5)
    # att = torch.nn.functional.softmax(att, dim=2).unsqueeze(1)
    # return att  # att bx1x5x18

    att = []
    for q_em, da_em in zip(batch_q_em, batch_da_em):
        D = q_em.size()[1]
        T_batch_da_em = torch.transpose(da_em, 0, 1)
        temp_att = torch.matmul(q_em, T_batch_da_em)
        temp_att = temp_att / (D ** 0.5)
        temp_att = torch.nn.functional.sigmoid(temp_att).unsqueeze(0)
        att.append(temp_att)
    return att  # att bx1x5x18


class NTN(torch.nn.Module):
    def __init__(self, D, k):
        super(NTN, self).__init__()
        self.k = k
        self.D = D
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.w = torch.nn.Parameter(torch.Tensor(self.k, self.D, self.D))
        self.V = torch.nn.Parameter(torch.Tensor(self.k, 2 * self.D))
        self.b = torch.nn.Parameter(torch.Tensor(self.k, 1, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.w)
        torch.nn.init.xavier_uniform_(self.V)
        torch.nn.init.xavier_uniform_(self.b)

    def forward(self, embed_q, embed_da):  # batch_q_em bx5xc   batch_da_em bx18xc   torch.tensor
        out = []
        for e_q, e_da in zip(embed_q, embed_da):
            q_size = e_q.size(0)
            da_size = e_da.size(0)
            
            batch_q_em_adddim = torch.unsqueeze(e_q, 0)  # batch_q_em_adddim bx1x5xc   torch.tensor
            batch_da_em_adddim = torch.unsqueeze(e_da, 0)  # batch_da_em _adddim bx1x18xc   torch.tensor
            T_batch_da_em_adddim = torch.transpose(batch_da_em_adddim, 1,
                                                   2)  # T_batch_da_em _adddim bx1xcx18   torch.tensor
            # first part
            first = torch.matmul(batch_q_em_adddim, self.w)  # first bxkx5xc   torch.tensor
            first = torch.matmul(first, T_batch_da_em_adddim)  # first bxkx5x18   torch.tensor
            # first part
            # second part
            ed_batch_q_em = torch.unsqueeze(e_q, 1)  # ed_batch_q_em bx5x1xc   torch.tensor
            ed_batch_q_em = ed_batch_q_em.repeat(1, da_size, 1)  # ed_batch_q_em bx5x18xc   torch.tensor
            ed_batch_q_em = ed_batch_q_em.reshape(q_size * da_size, self.D)  # ed_batch_q_em bx90xc
    
            ed_batch_da_em = torch.unsqueeze(e_da, 0)  # ed_batch_da_em bx1x18xc   torch.tensor
            ed_batch_da_em = ed_batch_da_em.repeat(q_size, 1, 1)  # ed_batch_da_em bx5x18xc   torch.tensor
            ed_batch_da_em = ed_batch_da_em.reshape(q_size * da_size, self.D)  # ed_batch_da_em bx90xc
    
            mid = torch.cat([ed_batch_q_em, ed_batch_da_em], 1)  # mid bx90x2c
            mid = torch.transpose(mid, 0, 1)  # mid bx2cx90
            mid = torch.matmul(self.V, mid)  # mid bxkx90
            mid = mid.reshape(self.k, q_size, da_size)  # mid bxkx5x18
            # second part
            end = first + mid + self.b
            out.append(torch.sigmoid(end))
        return out  # end bxkx5x18

