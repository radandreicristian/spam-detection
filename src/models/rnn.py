from typing import Any, Optional

import torch
import torch.nn as nn
from src.models.base_model import BaseSpamClassifier
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.optim import AdamW

from src.models.embedders import BaseEmbedder


class RnnSpamClassifier(BaseSpamClassifier):
    def __init__(self,
                 *args,
                 **kwargs):
        super(RnnSpamClassifier, self).__init__()
        hyper_cfg = kwargs.get("hyper", {})

        self.activation = hyper_cfg.get("activation", None)
        self.bidirectional = hyper_cfg.get("bidirectional", False)
        self.bidirectional_multiplier = 2 if self.bidirectional else 1
        self.n_layers = hyper_cfg.get("n_layers")
        self.p_dropout = hyper_cfg.get("p_dropout")
        self.d_hidden = hyper_cfg.get("d_hidden")
        self.d_input = hyper_cfg.get("d_input")
        self.learning_rate = hyper_cfg.get("learning_rate")
        self.clip_value = hyper_cfg.get("clip_value")

        embedder: BaseEmbedder = kwargs.get("embedder")

        weights = torch.tensor(embedder.get_weights(), dtype=torch.float)

        n_embeddings, d_embedding = weights.shape

        self.embedding = nn.Embedding(num_embeddings=n_embeddings,
                                      embedding_dim=d_embedding,
                                      _weight=weights)

        # Freeze embedding layer
        for param in self.embedding.parameters():
            param.requires_grad = False

        self.rnn = nn.RNN(input_size=self.d_input,
                          hidden_size=self.d_hidden,
                          num_layers=self.n_layers,
                          nonlinearity=self.activation,
                          dropout=self.p_dropout,
                          bidirectional=self.bidirectional,
                          batch_first=True)

        self.d_hidden_mlp = self.d_hidden * self.bidirectional_multiplier

        self.mlp = nn.ModuleList([nn.Linear(in_features=self.d_hidden_mlp,
                                            out_features=1)])

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        # https://stackoverflow.com/a/54816498/10831784
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -self.clip_value, self.clip_value))

    def forward(self, x):
        # x (batch_size, seq_len, d_embedding)
        embeddings = self.embedding(x)

        lengths = torch.tensor(list(map(len, x)))

        packed_embeddings = pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=True)

        # features (batch_size, seq_len, D * d_hidden), D=2 if bidirectional, 1 otherwise
        rnn_out, _ = self.rnn(x)

        output, sizes = pad_packed_sequence(rnn_out, batch_first=True)
        # take only the output of the last cell
        # features (batch_size, 1, D * d_hidden)

        last_seq = [output[e, i - 1, :].unsqueeze(0) for e, i in enumerate(sizes)]
        features = torch.cat(last_seq, dim=0)

        features = features[:, -1, :]
        for mlp_layer in self.mlp:
            features = mlp_layer(features)

        # features (batch_size, 1, 1)
        features = torch.flatten(features)

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
