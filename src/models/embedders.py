import logging
from abc import ABC, abstractmethod
from typing import List

import gensim
import torch

from src.models.padding import pad_tensor

logger = logging.getLogger()
logger.setLevel(logging.WARNING)


class BaseEmbedder(ABC):

    @abstractmethod
    def embed(self,
              sequence: List[str]) -> torch.Tensor:
        pass

    @abstractmethod
    def pad(self,
            sequence: torch.Tensor) -> torch.Tensor:
        pass


class FastTextEmbedder(BaseEmbedder):

    def __init__(self,
                 **kwargs):
        embeddings_path = kwargs.get("path")
        self.max_seq_len = kwargs.get("max_seq_len")
        self.embeddings = gensim.models.keyedvectors.load_word2vec_format(embeddings_path, binary=False)

    def embed(self,
              sequence: str) -> torch.Tensor:
        split_sequence = sequence.split()
        seq_len = len(split_sequence)

        d_embedding = len(self.embeddings.get_vector("anything"))
        sequence_embedding = torch.zeros(size=(seq_len, d_embedding))
        for index, token in enumerate(split_sequence):
            try:
                embedding = torch.tensor(self.embeddings.get_vector(token),
                                         dtype=torch.float32)
                sequence_embedding[index, :] = embedding
            except KeyError:
                logger.debug(f"Not found in dictionary: {token}")
        return sequence_embedding

    def pad(self,
            sequence: torch.Tensor) -> torch.Tensor:
        return pad_tensor(sequence=sequence,
                          max_seq_len=self.max_seq_len)


class EmbedderFactory:

    @staticmethod
    def get_embedder(*args,
                     **kwargs) -> BaseEmbedder:
        tag = kwargs.get("tag", "")
        if tag == "fasttext":
            return FastTextEmbedder(**kwargs)
        else:
            raise ValueError('Invalid embedder name.')
