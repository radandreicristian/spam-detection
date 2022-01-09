from typing import Any, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


class SpamDataset(Dataset):
    def __init__(self,
                 data: pd.DataFrame,
                 embedder) -> None:
        self.data = data
        self.embedder = embedder

    def __len__(self):
        return len(self.data)

    def __getitem__(self,
                    item_idx: int) -> dict:
        item = self.data.iloc[item_idx]
        sequence = item["content"]
        label = torch.tensor(item["label"], dtype=torch.float32)

        embeddings = self.embedder.embed(sequence)
        padded_embeddings = self.embedder.pad(embeddings)

        return {"embeddings": padded_embeddings,
                "labels": label}
