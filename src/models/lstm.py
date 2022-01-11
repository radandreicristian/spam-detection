from typing import Any, Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam
from torchmetrics import F1, Recall
from models.base_model import BaseSpamClassifier

from models.embedders import BaseEmbedder


class LstmSpamClassifier(BaseSpamClassifier):
    def __init__(self,
                 *args,
                 **kwargs):
        super(LstmSpamClassifier, self).__init__()
        hyper_cfg = kwargs.get("hyper", {})

        self.bidirectional = hyper_cfg.get("bidirectional")

        self.bidirectional_multiplier = 2 if self.bidirectional else 1
        self.n_layers = hyper_cfg.get("n_layers")
        self.p_dropout = hyper_cfg.get("p_dropout")
        self.d_hidden = hyper_cfg.get("d_hidden")
        self.d_input = hyper_cfg.get("d_input")
        self.learning_rate = hyper_cfg.get("learning_rate")

        embedder: BaseEmbedder = kwargs.get("embedder")

        weights = torch.tensor(embedder.get_weights(), dtype=torch.float)

        n_embeddings, d_embedding = weights.shape

        self.embedding = nn.Embedding(num_embeddings=n_embeddings,
                                      embedding_dim=d_embedding,
                                      _weight=weights)

        self.lstm = nn.LSTM(input_size=self.d_input,
                            hidden_size=self.d_hidden,
                            num_layers=self.n_layers,
                            batch_first=True,
                            dropout=self.p_dropout,
                            bidirectional=self.bidirectional)

        self.d_hidden_mlp = self.d_hidden * self.bidirectional_multiplier
        """
        self.mlp = nn.ModuleList([
            nn.Linear(in_features=self.d_hidden_mlp,
                      out_features=self.d_hidden_mlp // 2),
            nn.ReLU(),
            nn.Dropout(self.p_dropout),
            nn.Linear(in_features=self.d_hidden_mlp // 2,
                      out_features=1)
        ])
        """
        self.mlp = nn.ModuleList([nn.Linear(in_features=self.d_hidden_mlp,
                                            out_features=1)])

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        self.f1 = F1(num_classes=1).to(self.device)
        self.recall = Recall(num_classes=1).to(self.device)

        self.metrics = {"f1": self.f1,
                        "recall": self.recall}

    def forward(self,
                x: torch.Tensor,
                *args,
                **kwargs) -> torch.Tensor:
        embeddings = self.embedding(x)

        lengths = torch.tensor(list(map(len, x)))

        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=True)

        # features (batch_size, max_seq_len, D * d_hidden)
        lstm_out, (_, _) = self.lstm(packed_embeddings)

        output, sizes = pad_packed_sequence(lstm_out, batch_first=True)

        # https://discuss.pytorch.org/t/get-each-sequences-last-item-from-packed-sequence/41118/3
        last_seq = [output[e, i - 1, :].unsqueeze(0) for e, i in enumerate(sizes)]
        features = torch.cat(last_seq, dim=0)
        # features (batch_size, 1, D * d_hidden)
        # features = lstm_out[:, -1, :]

        for mlp_layer in self.mlp:
            features = mlp_layer(features)

        # features (batch_size, 1, 1)
        features = torch.flatten(features)

        # features (batch_size,)
        prediction = self.sigmoid(features)

        return prediction

    def common_step(self,
                    batch: Any,
                    batch_idx: int) -> STEP_OUTPUT:
        sequence, labels = batch

        predictions = self(sequence)

        loss = self.criterion(predictions, labels)

        int_labels = labels.int().to(self.device)
        self.update_all_metrics(predictions=predictions,
                                labels=int_labels)

        return {"loss": loss,
                "metrics": self.metrics}

    def training_step(self,
                      batch: Any,
                      batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        loss, metrics = self.common_step(batch, batch_idx).values()

        self.log("train_loss", loss)
        # self.log("train_loss", self.train_loss, on_epoch=True)
        return loss

    def validation_step(self,
                        batch: Any,
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, metrics = self.common_step(batch, batch_idx).values()
        self.log("valid_loss", loss, on_epoch=True)
        return loss

    def test_step(self,
                  batch: Any,
                  batch_idx: int) -> Optional[STEP_OUTPUT]:
        _, metrics = self.common_step(batch, batch_idx)
        self.log_dict(metrics)

        return metrics

    def predict_step(self,
                     batch: Any,
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None) -> Any:
        embeddings = batch['embeddings']
        predictions = self(embeddings)
        return predictions

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(),
                         lr=self.learning_rate)
        return optimizer
