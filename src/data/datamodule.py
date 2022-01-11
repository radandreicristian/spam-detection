from typing import Optional

import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, WeightedRandomSampler

import pytorch_lightning as pl
import pandas as pd

from src.data.datasets import SpamDataset
from src.data.sampler import get_weighted_sampler


class SpamDataModule(pl.LightningDataModule):
    def __init__(self,
                 paths: list,
                 train_batch_size,
                 word2idx):
        super(SpamDataModule, self).__init__()
        train_path, valid_path, test_path = paths

        self.train_df = pd.read_csv(train_path, index_col=False)
        self.batch_size = train_batch_size
        self.valid_df = pd.read_csv(valid_path, index_col=False)
        self.test_df = pd.read_csv(test_path, index_col=False)

        self.train_sampler: Optional[WeightedRandomSampler] = None
        self.valid_sampler: Optional[WeightedRandomSampler] = None
        self.train_dataset: Optional[SpamDataset] = None
        self.valid_dataset: Optional[SpamDataset] = None
        self.test_dataset: Optional[SpamDataset] = None

        self.word2idx = word2idx

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SpamDataset(data=self.train_df,
                                         word2idx=self.word2idx)
        self.valid_dataset = SpamDataset(data=self.valid_df,
                                         word2idx=self.word2idx)
        self.test_dataset = SpamDataset(data=self.test_df,
                                        word2idx=self.word2idx)

        self.train_sampler = get_weighted_sampler(self.train_df["label"])
        self.valid_sampler = get_weighted_sampler(self.valid_df["label"])

    def pad_collate(self, batch):
        x, y = zip(*batch)

        x = [torch.tensor(el, dtype=torch.int) for el in x]

        x_pad = pad_sequence(x, batch_first=True, padding_value=0)

        y = torch.tensor(y)

        return x_pad, y

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          sampler=self.train_sampler,
                          collate_fn=self.pad_collate)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_dataset,
                          batch_size=self.batch_size,
                          num_workers=8,
                          sampler=self.valid_sampler,
                          collate_fn=self.pad_collate)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset,
                          batch_size=1,
                          num_workers=8,
                          collate_fn=self.pad_collate)
