from typing import Optional, Sequence

import numpy as np
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, Dataset

from traffic_signs_dataset import TraficSignDataset


class TrafficSignDataModule(LightningDataModule):
    """Datamodule which does something..."""

    def __init__(
            self,
            data_dir: str = "data/",
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            train_val_test_percentage_split: Sequence[int] = (0.70, 0.10, 0.20),
            seed=42
    ):
        """
        Args:
            data_dir: Path to folder with data.
            batch_size: Batch size.
            num_workers: Number of processes for data loading.
            pin_memory: Whether to pin CUDA memory (slight speed up for GPU users).
            ...
        """
        super().__init__()

        self.seed = seed
        self.train_test_val_percentage_split = train_val_test_percentage_split
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.pre_transforms = None
        self.transforms = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.node_feature_dim = None
        self.edge_feature_dim = None

    def prepare_data(self):
        TraficSignDataset(root=self.data_dir)

    def setup(self, stage: Optional[str] = None):
        """Load data. Split data. Set variables: self.data_train, self.data_val, self.data_test."""
        if not self.data_train or not self.data_val or not self.data_test:
            dataset = TraficSignDataset(root=self.data_dir)
            self.data_train, self.data_val, self.data_test = random_split(dataset, self.train_test_val_percentage_split, torch.Generator().manual_seed(self.seed))

            self.node_feature_dim = dataset[0].x_d[0].shape[-1]
            if dataset[0].edge_features_d[0] is not None:
                self.edge_feature_dim = dataset[0].edge_features_d[0].shape[-1]
            else:
                self.edge_feature_dim = 0

    def __len__(self):
        return 7218

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            follow_batch=["x_d", "x_q"],
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            follow_batch=["x_d", "x_q"],
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            follow_batch=["x_d", "x_q"],
        )
