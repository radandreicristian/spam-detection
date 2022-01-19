import logging
from abc import ABC, abstractmethod
from typing import List, Dict

import fasttext
import fasttext.util
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.WARNING)

from copy import deepcopy


class BaseEmbedder(ABC):

    @abstractmethod
    def get_weights(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_vocab(self) -> List:
        pass

    @abstractmethod
    def get_word2idx(self) -> Dict:
        pass


class FastTextEmbedder(BaseEmbedder):

    def __init__(self,
                 *args,
                 **kwargs):
        embeddings_path = kwargs.get("path")
        pretrained_embeddings = fasttext.load_model(embeddings_path)
        d_embedding_pretrained = pretrained_embeddings.get_dimension()
        d_embedding = kwargs.get("dimension")
        pretrained_embeddings = fasttext.util.reduce_model(pretrained_embeddings, d_embedding) \
            if d_embedding_pretrained != d_embedding else pretrained_embeddings

        vocab = kwargs.get("vocab")
        word2idx = kwargs.get("word2idx")

        weights = np.empty((len(vocab), d_embedding))

        # Initialize vectors for "" and "UNK"
        weights[0] = np.zeros(d_embedding, dtype='float32')
        weights[1] = np.random.uniform(-.25, .25, d_embedding)

        word2idx_copy = deepcopy(word2idx)
        word2idx_copy.pop("")
        word2idx_copy.pop("UNK")
        for word, idx in word2idx_copy.items():
            embedding = pretrained_embeddings[word]
            weights[idx] = embedding

        self.vocab = vocab
        self.word2idx = word2idx
        self.weights = weights

    def get_weights(self):
        return self.weights

    def get_vocab(self) -> List:
        return self.vocab

    def get_word2idx(self) -> Dict:
        return self.word2idx


class GloveEmbedder(BaseEmbedder):

    def __init__(self,
                 *args,
                 **kwargs):
        self.embeddings_path = kwargs.get("path")
        self.embedding_index = {}
        self.read_embeddings()
        self.d_embedding = kwargs.get("dimension")

        vocab = kwargs.get("vocab")
        word2idx = kwargs.get("word2idx")

        weights = np.empty((len(vocab), self.d_embedding))

        # Initialize vectors for "" and "UNK"
        weights[0] = np.zeros(self.d_embedding, dtype='float32')
        weights[1] = np.random.uniform(-.25, .25, self.d_embedding)

        word2idx_copy = deepcopy(word2idx)
        word2idx_copy.pop("")
        word2idx_copy.pop("UNK")

        glove_vocab = self.embedding_index.keys()
        for word, idx in word2idx_copy.items():
            if word in glove_vocab:
                embedding = self.embedding_index[word]
            else:
                embedding = weights[1]
            weights[idx] = embedding

        self.vocab = vocab
        self.word2idx = word2idx
        self.weights = weights

    def get_weights(self):
        return self.weights

    def get_vocab(self) -> List:
        return self.vocab

    def get_word2idx(self) -> Dict:
        return self.word2idx

    def read_embeddings(self):
        f = open(self.embeddings_path, encoding="utf-8")
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.embedding_index[word] = coefs
        f.close()


class TrainableEmbedder(BaseEmbedder):

    def __init__(self,
                 *args,
                 **kwargs):
        self.embeddings_path = kwargs.get("path")
        self.d_embedding = kwargs.get("dimension")

        vocab = kwargs.get("vocab")
        word2idx = kwargs.get("word2idx")

        weights = np.empty((len(vocab), self.d_embedding))

        # Initialize vectors for "" and "UNK"
        weights[0] = np.zeros(self.d_embedding, dtype='float32')
        weights[1] = np.random.uniform(-.25, .25, self.d_embedding)

        word2idx_copy = deepcopy(word2idx)
        word2idx_copy.pop("")
        word2idx_copy.pop("UNK")

        for word, idx in word2idx_copy.items():
            weights[idx] = np.random.uniform(-.25, .25, self.d_embedding)

        self.vocab = vocab
        self.word2idx = word2idx
        self.weights = weights

    def get_weights(self):
        return self.weights

    def get_vocab(self) -> List:
        return self.vocab

    def get_word2idx(self) -> Dict:
        return self.word2idx


class EmbedderFactory:

    @staticmethod
    def get_embedder(vocab,
                     word2idx,
                     *args,
                     **kwargs) -> BaseEmbedder:
        embedding_type = kwargs.get("used_embeddings")
        embedding_type_cfg = kwargs.get(embedding_type, {})

        kwargs = {'vocab': vocab,
                  'word2idx': word2idx,
                  **embedding_type_cfg}
        if embedding_type == "pretrained_fasttext":
            return FastTextEmbedder(**kwargs)
        if embedding_type == "glove":
            return GloveEmbedder(**kwargs)
        if embedding_type == "trainable":
            return TrainableEmbedder(**kwargs)
        else:
            raise ValueError('Invalid embedder name.')
