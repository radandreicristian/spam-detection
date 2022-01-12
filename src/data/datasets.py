from typing import Any, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    """A custom Dataset class for the spam detection data."""
    def __init__(self,
                 data: pd.DataFrame,
                 word2idx: dict) -> None:
        """
        Initializes a SpamDataset object.

        :param data: A Pandas DataFrame object.
        :param word2idx: A dictionary mapping words in the vocabulary to unique indices.
        """
        self.data = data
        self.word2idx = word2idx

    def __len__(self):
        """
        Returns the length of the dataset (nr. of samples).

        :return: The length of the dataset.
        """
        return len(self.data)

    def __getitem__(self,
                    item_idx: int) -> Tuple:
        """
        Returns an item of the dataset, based on an index.

        :param item_idx: The index for retrieval.
        :return: A tuple consisting of the sentence (list containing index of each word) and the label.
        """
        item = self.data.iloc[item_idx]
        sequence = item["content"].split()
        label = torch.tensor(item["label"], dtype=torch.float)

        sequence = list(map(lambda x: self.word2idx.get(x, 1), sequence))

        return sequence, label
