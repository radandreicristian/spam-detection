from typing import Optional

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import DataLoader, WeightedRandomSampler

import pytorch_lightning as pl
import pandas as pd

from src.data.datasets import SpamDataset
from src.data.sampler import get_weighted_sampler


class SpamDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_path,
                 valid_path,
                 test_path,
                 train_batch_size,
                 embedder):
        super(SpamDataModule, self).__init__()
        self.train_df = pd.read_csv(train_path, index_col=False)
        self.batch_size = train_batch_size
        self.valid_df = pd.read_csv(valid_path, index_col=False)
        self.test_df = pd.read_csv(test_path, index_col=False)

        self.train_sampler: Optional[WeightedRandomSampler] = None
        self.valid_sampler: Optional[WeightedRandomSampler] = None
        self.train_dataset: Optional[SpamDataset] = None
        self.valid_dataset: Optional[SpamDataset] = None
        self.test_dataset: Optional[SpamDataset] = None

        self.embedder = embedder

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SpamDataset(data=self.train_df,
                                         embedder=self.embedder)
        self.valid_dataset = SpamDataset(data=self.valid_df,
                                         embedder=self.embedder)
        self.test_dataset = SpamDataset(data=self.test_df,
                                        embedder=self.embedder)

        self.train_sampler = get_weighted_sampler(self.train_df["label"])
        self.valid_sampler = get_weighted_sampler(self.valid_df["label"])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          sampler=self.train_sampler)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          sampler=self.valid_sampler)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          num_workers=8)
