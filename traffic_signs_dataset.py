import collections
import gc
import os
import xml.etree.ElementTree as et
from distutils.dir_util import copy_tree, remove_tree
from glob import glob
from os.path import basename, join, splitext

import numpy as np
import time
import torch
from memory_profiler import profile
from torch_geometric.data import InMemoryDataset, download_url
from torch_geometric.data.data import Data
from tqdm import tqdm
import utils
import torchvision.transforms

from PairData import PairData


class TraficSignDataset(InMemoryDataset):
    """Loads all graphs from the Trafic sign dataset"""

    def __init__(self, root="data/", data_graph_seg_size=100, query_graph_seg_size=20, data_graph_compactness=50,
                 query_graph_compactness=150, crop_size=(400, 400)):
        self.data_dir = root
        self.test_path = join(root, "test")
        self.train_path = join(root, "train")
        self.root = root
        self.crop_size = crop_size
        self.data_graph_seg_size = data_graph_seg_size
        self.query_graph_seg_size = query_graph_seg_size
        self.data_graph_compactness = data_graph_compactness
        self.query_graph_compactness = query_graph_compactness
        super().__init__(self.data_dir, pre_filter=None, pre_transform=None, transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    # def download(self):
    #     print("downloading AIDS dataset")
    #     download_url(self.url, self.raw_dir)
    #     extract_zip(join(self.raw_dir, self.raw_filename), self.raw_dir)
    #     copy_tree(join(self.raw_dir, "AIDS", "data"), join(self.raw_dir, "data"))
    #     remove_tree(join(self.raw_dir, "AIDS"))
    #     remove_tree(join(self.raw_dir, "__MACOSX"))

    @property
    def raw_file_names(self):
        return "data"

    @property
    def processed_file_names(self):
        return f"data_pair_qg{self.query_graph_seg_size}_size{self.crop_size[0]}.pt"

    def _generate_graphs(self, cropped_img):
        cropped_image, cropped_mask, category = cropped_img
        mark, mask = utils.get_mark(category)

        torch_image = torchvision.transforms.ToTensor()(cropped_image)
        torch_mark = torchvision.transforms.ToTensor()(mark)

        # print("image_segmentation start")
        #start = time.time()

        torch_seg_image_with_edges = utils.get_graph_from_image(torch_image, self.data_graph_seg_size,
                                                                self.data_graph_compactness, )

        #end = time.time()
        # print("image_segmentation run time", end - start)

        # print("mask_segmentation start")
        #start = time.time()
        torch_seg_mark_with_edges = utils.get_graph_from_image(torch_mark, self.query_graph_seg_size,
                                                               self.query_graph_compactness, mask)
        #end = time.time()
        # print("mask_segmentation run time", end - start)

        # print("truth matrix start")
        #start = time.time()
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
        #end = time.time()
        # print("truth matrix run time", end - start)
        return torch_seg_image_with_edges, torch_seg_mark_with_edges, t_m, category

    def process(self):
        img_idx = []

        final_data_list = []

        for file in glob((join(self.test_path, "*.jpg"))):
            img_idx.append(os.path.basename(file).split(".")[0])
        for file in glob((join(self.train_path, "*.jpg"))):
            img_idx.append(os.path.basename(file).split(".")[0])

        for img_id in tqdm(img_idx,
                           desc="Generating Traffic sign dataset",
                           colour="GREEN", ):

            try:
                cropped_img = utils.get_cropped_img_mask_cat(img_id, self.crop_size)
            except:
                print("cropping", img_id)
                continue

            if cropped_img is None:
                continue
            try:
                torch_seg_image_with_edges, torch_seg_mark_with_edges, t_m, category = self._generate_graphs(cropped_img)
            except:
                print("generate", img_id)
                continue
            final_data_list.append(
                pair_data_builder(torch_seg_image_with_edges, torch_seg_mark_with_edges, t_m, img_id, category))

        # ==========================================================================================

        np.random.shuffle(final_data_list)
        data, slices = self.collate(final_data_list)
        torch.save((data, slices), self.processed_paths[0], pickle_protocol=5)


def pair_data_builder(data_graph: Data, query_graph: Data, truth_matrix: torch.Tensor, d_id: str, category: str, ) -> PairData:
    return PairData(
        d_id=d_id,
        edge_features_d=data_graph.edge_attr,
        edge_index_d=data_graph.edge_index,
        x_d=data_graph.x,
        pos_d=data_graph.pos,
        edge_features_q=query_graph.edge_attr,
        edge_index_q=query_graph.edge_index,
        x_q=query_graph.x,
        pos_q=query_graph.pos,
        category=category,
        truth_matrix=truth_matrix
    )
