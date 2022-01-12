from typing import Any, Optional

import torch
import torch.nn as nn
from torchmetrics import F1

from src.models.base_model import BaseSpamClassifier
from src.models.embedders import BaseEmbedder
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim import Adam, AdamW


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

        # Freeze embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(input_size=self.d_input,
                            hidden_size=self.d_hidden,
                            num_layers=self.n_layers,
                            batch_first=True,
                            dropout=self.p_dropout,
                            bidirectional=self.bidirectional)

        self.d_hidden_mlp = self.d_hidden * self.bidirectional_multiplier

        self.mlp = nn.ModuleList([nn.Linear(in_features=self.d_hidden_mlp,
                                            out_features=1)])

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

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

        labels = labels.int().to(self.device)

        return {"loss": loss,
                "predicted": predictions,
                "labels": labels}

    def training_step(self,
                      batch: Any,
                      batch_idx: int) -> STEP_OUTPUT:  # type: ignore
        loss, predicted, labels = self.common_step(batch, batch_idx).values()

        self.train_f1(predicted, labels)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=False)

        self.log("train_loss", loss)
        return loss

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log("train_epoch_f1", self.train_f1)

    def validation_step(self,
                        batch: Any,
                        batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, predicted, labels = self.common_step(batch, batch_idx).values()

        self.valid_f1(predicted, labels)
        self.log("valid_f1", self.valid_f1, on_step=True, on_epoch=False)

        self.log("valid_loss", loss, on_epoch=True)
        return loss

    def validation_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        self.log("valid_epoch_f1", self.valid_f1)

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
        optimizer = AdamW(self.parameters(),
                          lr=self.learning_rate)
        return optimizer
