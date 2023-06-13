import os.path
import urllib

import glob
import numpy as np
import pandas as pd
import requests
import torch
import torch_geometric.data
from memory_profiler import profile
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
from skimage.segmentation import slic
from skimage.util import img_as_float
from skimage import io, transform, graph
from skimage.transform import AffineTransform, warp

from python import anno_func
import matplotlib.pyplot as plt
import torchvision.transforms
import torch.nn.functional as Fn
from torchvision.io import read_image
import networkx as nx
import torch
from torch import Tensor
import gc
from torch_geometric.data import Data
from torch_geometric.data.datapipes import functional_transform
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import scatter
from cuda_slic import slic as c_slic

NP_TORCH_FLOAT_DTYPE = np.float32
NP_TORCH_LONG_DTYPE = np.int64

NUM_FEATURES = 3

base_path = 'data'
ann_path = 'data/annotations.json'
os.makedirs(base_path, exist_ok=True)

marks_path = f'{base_path}/marks/pad-all/'
train_path = f'{base_path}/train/'
test_path = f'{base_path}/test/'

train_ids = open(f"{base_path}/train/ids.txt").read().splitlines()
test_ids = open(f"{base_path}/test/ids.txt").read().splitlines()

annos = json.loads(open(ann_path).read())


def get_cropped_img_mask_cat(imgid):
    img = annos["imgs"][imgid]
    imgpath = base_path + '/' + img['path']

    marks = list(glob.glob(f"{marks_path}*"))
    marks = [os.path.basename(x)[:-4] for x in marks]
    image = img_as_float(io.imread(imgpath))
    category = anno_func.load_img_single_category(annos, imgid, marks)
    if category is None:
        return None

    mask, center, height = anno_func.load_mask(annos, category, imgid, image)
    # plt.imshow(anno_func.draw_all(annos, category, imgid, image, have_label=False))
    # plt.show()

    scale_factor = 280 / height
    size = (1000, 1000)

    o_height = image.shape[0]
    o_width = image.shape[1]

    half_height = int(height)

    dist_to_left = center[0] - half_height
    dist_to_right = image.shape[0] - center[0] - half_height
    dist_to_top = center[1] - half_height
    dist_to_bottom = image.shape[1] - center[1] - half_height

    if dist_to_left < 0 or dist_to_right < 0 or dist_to_bottom < 0 or dist_to_top < 0:
        minimum = min(dist_to_top, dist_to_bottom, dist_to_right, dist_to_left)
        dist_to_right -= minimum
        dist_to_bottom -= minimum
        dist_to_left -= minimum
        dist_to_top -= minimum

    # Obliczenie nowych granic przycięcia
    left = dist_to_left - int(dist_to_left / scale_factor)
    right = int(o_width - dist_to_right + (dist_to_right / scale_factor))
    top = dist_to_top - int(dist_to_top / scale_factor)
    bottom = int(o_height - dist_to_bottom + (dist_to_bottom / scale_factor))


    cropped_image = image[top:bottom, left:right]
    cropped_mask = mask[top:bottom, left:right]


    rescaled_image = transform.resize(cropped_image, size, anti_aliasing=True)
    rescaled_mask = transform.resize(np.asarray(cropped_mask, dtype=bool), size, anti_aliasing=None)


    # Przycięcie obrazu

    return rescaled_image, rescaled_mask, category


def get_mark(mark_name):
    imgpath = f"{marks_path}{mark_name}.png"
    image = img_as_float(io.imread(imgpath))
    image = np.asarray(image)
    image[:, :, 3] = np.ceil(image[:, :, 3])
    return image[:, :, :3], image[:, :, 3]


def autocrop_image(image):
    image_data = np.asarray(image)
    image_data_bw = image_data[:, :, 3]
    non_empty_columns = np.where(image_data_bw.max(axis=0) > 0)[0]
    non_empty_rows = np.where(image_data_bw.max(axis=1) > 0)[0]
    cropBox = (min(non_empty_rows), max(non_empty_rows),
               min(non_empty_columns), max(non_empty_columns))

    image_data_new = image_data[cropBox[0]:cropBox[
                                               1] + 1, cropBox[2]:cropBox[3] + 1, :]

    return image_data_new[:, :, 3]


def get_graph_from_image(image, n_segments, compactness, mask=None):
    # load the image and convert it to a floating point data type
    if mask is None:
        slic_graph = ToSLIC(n_segments=n_segments, compactness=compactness, add_seg=True, use_cuda=True)(image)
    else:
        slic_graph = ToSLIC(n_segments=n_segments, compactness=compactness, add_seg=True, mask=mask)(image)
    segments = slic_graph.seg
    segments = segments.permute(1, 2, 0)

    g = graph.rag_mean_color(image.permute(1, 2, 0).numpy(), segments.numpy())
    if mask is not None:
        g.remove_node(0)
        edges_indexes = [[x - 1 for x, y in g.edges], [y - 1 for x, y in g.edges]]
    else:
        edges_indexes = [[x for x, y in g.edges], [y for x, y in g.edges]]
    edges_indexes = torch.Tensor(edges_indexes)
    edges_indexes = torch_geometric.utils.add_self_loops(edges_indexes)

    # lc = graph.show_rag(segments.numpy().reshape(segments.shape[0], segments.shape[1]), g, image.permute(1, 2, 0).numpy(), border_color="blue")
    # figure = plt.gcf()  # get current figure
    # figure.set_size_inches(20, 20)
    #
    # plt.savefig('rag_full.png')
    if mask is not None:
        x = slic_graph.x[1:]
        pos = slic_graph.pos[1:]
    else:
        x = slic_graph.x
        pos = slic_graph.pos

    torch_graph = torch_geometric.data.Data(x=x, edge_index=edges_indexes[0], pos=pos)
    return torch_graph


def draw_graph(torch_graph, size: tuple, path, **kwargs):
    networkx_image_graph = torch_geometric.utils.to_networkx(torch_graph, to_undirected=True)

    options = {
        "node_size": 10,
        "edge_color": "black",
        "arrowstyle": "-",
        "arrows": False,
    }

    nx_position = {x: [y.numpy()[0], -y.numpy()[1]] for x, y in enumerate(torch_graph.pos)}
    networkx_image_graph.remove_edges_from(nx.selfloop_edges(networkx_image_graph))
    nx.draw(networkx_image_graph, nx_position, **options, **kwargs)

    figure = plt.gcf()  # get current figure
    figure.set_size_inches(*size)

    plt.savefig(path, dpi=100)
    plt.clf()


class ToSLIC(BaseTransform):
    def __init__(self, add_seg: bool = False, add_img: bool = False, use_cuda=False, **kwargs):
        self.add_seg = add_seg
        self.add_img = add_img
        self.use_cuda = use_cuda
        self.kwargs = kwargs

    def __call__(self, img: Tensor) -> Data:
        from skimage.segmentation import slic

        img = img.permute(1, 2, 0)
        h, w, c = img.size()

        if self.use_cuda:
            seg = c_slic(img.to(torch.double).numpy(), **self.kwargs)
        else:
            seg = slic(img.to(torch.double).numpy(), **self.kwargs)
        seg = torch.from_numpy(seg)

        x = scatter(img.view(h * w, c), seg.view(h * w), dim=0, reduce='mean')

        pos_y = torch.arange(h, dtype=torch.float)
        pos_y = pos_y.view(-1, 1).repeat(1, w).view(h * w)
        pos_x = torch.arange(w, dtype=torch.float)
        pos_x = pos_x.view(1, -1).repeat(h, 1).view(h * w)

        pos = torch.stack([pos_x, pos_y], dim=-1)
        pos = scatter(pos, seg.view(h * w), dim=0, reduce='mean')

        data = Data(x=x, pos=pos)

        if self.add_seg:
            data.seg = seg.view(1, h, w)

        if self.add_img:
            data.img = img.permute(2, 0, 1).view(1, c, h, w)

        return data


def euclidean_node_mapping(mask_array, mark_array):  # returns [mask_id, mark_id]
    result = []
    if len(mask_array) > len(mark_array):
        temp_mask_array = mask_array.copy()
        for mark_id, mark_pos in mark_array:
            min_distance = np.inf
            min_id = None
            min_index = None
            for index, [mask_id, mask_pos] in enumerate(temp_mask_array):
                distance = np.linalg.norm(np.array(mask_pos) - np.array(mark_pos))
                if distance < min_distance:
                    min_index = index
                    min_distance = distance
                    min_id = mask_id
            result.append([min_id, mark_id])
            del temp_mask_array[min_index]

        for mask_id, mask_pos in temp_mask_array:
            min_distance = np.inf
            min_id = None
            for mark_id, mark_pos in mark_array:
                distance = np.linalg.norm(np.array(mask_pos) - np.array(mark_pos))
                if distance < min_distance:
                    min_distance = distance
                    min_id = mark_id
            result.append([mask_id, min_id])
    else:
        temp_mark_array = mark_array.copy()
        for mask_id, mask_pos in mask_array:
            min_distance = np.inf
            min_id = None
            min_index = None
            for index, [mark_id, mark_pos] in enumerate(temp_mark_array):
                distance = np.linalg.norm(np.array(mask_pos) - np.array(mark_pos))
                if distance < min_distance:
                    min_index = index
                    min_distance = distance
                    min_id = mark_id
            result.append([mask_id, min_id])
            del temp_mark_array[min_index]

        for mark_id, mark_pos in temp_mark_array:
            min_distance = np.inf
            min_id = None
            for mask_id, mask_pos in mask_array:
                distance = np.linalg.norm(np.array(mask_pos) - np.array(mark_pos))
                if distance < min_distance:
                    min_distance = distance
                    min_id = mask_id
            result.append([min_id, mark_id])

    return result


def t_matrix(node_mapping, img_graph_size, mark_graph_size):  # node_mapping [mask_id, mark_id]
    t_m = np.zeros((img_graph_size, mark_graph_size))
    t_m[node_mapping[:, 0], node_mapping[:, 1]] = 1
    return t_m
