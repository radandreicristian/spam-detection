from typing import Any, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 word2idx: dict) -> None:
        self.data = data
        self.word2idx = word2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self,
                    item_idx: int) -> Tuple:
        item = self.data.iloc[item_idx]
        sequence = item["content"].split()
        label = torch.tensor(item["label"], dtype=torch.float)

        sequence = list(map(lambda x: self.word2idx.get(x, 1), sequence))

        return sequence, label
