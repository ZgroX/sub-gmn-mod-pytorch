import os.path
import urllib

import glob
import numpy as np
import pandas as pd
import requests
import torch
import torch_geometric.utils
import torchvision.transforms
from pytorch_lightning import LightningDataModule
from skimage.segmentation import mark_boundaries
from sklearn import datasets
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, FunctionTransformer
from torch.utils.data import DataLoader, Dataset, TensorDataset
import json
import pylab as pl
import random
import cv2
from torch_geometric import transforms as ts
from torch_geometric.transforms import ToSLIC
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io, transform, color
from skimage.transform import AffineTransform, warp

from PairData import PairData
from python import anno_func
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import torch.nn.functional as Fn
from torchvision.io import read_image
import utils
import networkx as nx
import collections
from traffic_signs_datamodule import TrafficSignDataModule


def ski_slic():
    imgid = '2'

    cropped_image, cropped_mask, category = utils.get_cropped_img_mask_cat(imgid)
    io.imsave("images/zimage.png", cropped_image)
    io.imsave("images/mask.png", cropped_mask)

    mark, mask = utils.get_mark(category)

    seg_image = slic(cropped_image, n_segments=10_000, compactness=10)
    print(f'SLIC number of image segments: {len(np.unique(seg_image))}')
    seg_mark = slic(mark, n_segments=100, compactness=10, mask=mask)
    print(f'SLIC number of mark segments: {len(np.unique(seg_mark))}')

    mean_image = color.label2rgb(seg_image, cropped_image, kind='avg')
    mean_mark = color.label2rgb(seg_mark, mark, kind='avg')
    # Wyświetl wynikową mapę superpikseli jako regiony
    io.imsave("images/mean_image.png", mark_boundaries(cropped_image, seg_image))
    io.imsave("images/mean_mark.png", mark_boundaries(mark, seg_mark))
    io.imsave("images/mark.png", mark)
    io.imsave("images/seg_mark.png", seg_mark)


def pytorch_slic():
    imgid = '23068'

    cropped_image, cropped_mask, category = utils.get_cropped_img_mask_cat(imgid)
    plt.imshow(cropped_image)
    plt.imshow(cropped_mask, alpha=0.8)
    plt.show()

    io.imsave("images/zimage.png", cropped_image)
    io.imsave("images/mask.png", cropped_mask)

    mark, mask = utils.get_mark(category)

    torch_image = torchvision.transforms.ToTensor()(cropped_image)
    torch_mark = torchvision.transforms.ToTensor()(mark)

    torch_seg_image_with_edges = utils.get_graph_from_image(torch_image, 100, 50)

    utils.draw_graph(torch_seg_image_with_edges, (20, 20), 'images/rag_graph.png')


def truth_matrix():
    imgid = '23068'

    cropped_image, cropped_mask, category = utils.get_cropped_img_mask_cat(imgid)
    mark, mask = utils.get_mark(category)

    torch_image = torchvision.transforms.ToTensor()(cropped_image)
    torch_mark = torchvision.transforms.ToTensor()(mark)

    torch_seg_image_with_edges = utils.get_graph_from_image(torch_image, 100, 50)
    torch_seg_mark_with_edges = utils.get_graph_from_image(torch_mark, 20, 150, mask)

    img_pos = np.round(torch_seg_image_with_edges.pos.numpy()).tolist()
    mask_pos = []
    mask_nodes_ids = []
    for y, row in enumerate(cropped_mask):
        for x, col in enumerate(row):
            if col:
                mask_pos.append([x, y])
    for i, pos in enumerate(img_pos):
        if pos in mask_pos:
            mask_nodes_ids.append(i)

    mark_pos = torch_seg_mark_with_edges.pos

    mark_pos[:, 0] -= mark_pos[:, 0].min()
    mark_pos[:, 1] -= mark_pos[:, 1].min()

    mark_ids = np.arange(len(mark_pos))

    zip_mark = [[a, [b, c]] for a, b, c in np.column_stack((mark_ids, mark_pos))]

    mask_nodes_ids = np.array(mask_nodes_ids)
    mask_pos = torch_seg_image_with_edges.pos[mask_nodes_ids]
    mask_pos[:, 0] -= mask_pos[:, 0].min()
    mask_pos[:, 1] -= mask_pos[:, 1].min()

    zip_mask = [[a, [b, c]] for a, b, c in np.column_stack((mask_nodes_ids, mask_pos))]

    mapped_nodes = utils.euclidean_node_mapping(zip_mask, zip_mark)
    mapped_nodes = np.array(mapped_nodes).astype(np.int16)
    t_m = utils.t_matrix(mapped_nodes, len(img_pos), len(mark_pos))

    mark_labels = {}
    i = 0
    for node in range(len(mark_pos)):
        if node in mapped_nodes[:, 1]:
            mark_labels[node] = f"{i}"
            i += 1
        else:
            mark_labels[node] = "X"

    mask_labels = {}
    for node in range(len(img_pos)):
        if node in mapped_nodes[:, 0]:
            mask_labels[node] = mark_labels[mapped_nodes[mapped_nodes[:, 0] == node, 1][0]]
        else:
            mask_labels[node] = "X"

    node_color = np.full(len(img_pos), "blue")
    node_color[mask_nodes_ids] = "red"

    utils.draw_graph(torch_seg_image_with_edges, (60, 60), 'images/masked_graph_red.png', with_labels=True, labels=mask_labels,
                     node_color=node_color)

    utils.draw_graph(torch_seg_mark_with_edges, (20, 20), 'images/masked_mark_graph_red.png', with_labels=True,
                     labels=mark_labels)



def draw_graphs_from_pair(pair: PairData):

    query_graph = torch_geometric.data.Data(x=pair.x_q[0], edge_index=pair.edge_index_q[0], pos=pair.pos_q[0])
    data_graph = torch_geometric.data.Data(x=pair.x_d[0], edge_index=pair.edge_index_d[0], pos=pair.pos_d[0])
    t_m = pair.truth_matrix



    img_labels = {}
    i = 0
    for n_id, node in enumerate(t_m[:, 0]):
        if torch.sum(node) > 0:
            img_labels[n_id] = f"{i}"
            i += 1
        else:
            img_labels[n_id] = "X"
    #
    # mask_labels = {}
    # for node in t_m[:, 1]:
    #     if node in mapped_nodes[:, 0]:
    #         mask_labels[node] = mark_labels[mapped_nodes[mapped_nodes[:, 0] == node, 1][0]]
    #     else:
    #         mask_labels[node] = "X"
    #
    # node_color = np.full(len(data_graph.pos), "blue")
    # node_color[mask_nodes_ids] = "red"
    #
    # utils.draw_graph(data_graph, (60, 60), 'images/masked_graph_red.png', with_labels=True,
    #                  labels=mask_labels,
    #                  node_color=node_color)
    #
    # utils.draw_graph(query_graph, (20, 20), 'images/masked_mark_graph_red.png', with_labels=True,
    #                  labels=mark_labels)


if __name__ == '__main__':
    # truth_matrix()
    pytorch_slic()
    # data_module = TrafficSignDataModule(batch_size=1)
    # data_module.prepare_data()
    # data_module.setup()
    # draw_graphs_from_pair(data_module.data_train[0])
    # x=0

