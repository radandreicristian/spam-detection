import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


def get_weighted_sampler(labels: pd.Series) -> WeightedRandomSampler:
    encoded_labels = labels.to_numpy()
    counts = np.array([len(np.where(encoded_labels == t)[0]) for t in np.unique(encoded_labels)])

    weights = 1. / counts
    sample_weights = torch.from_numpy(np.array([weights[t] for t in encoded_labels])).float()

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    return sampler
