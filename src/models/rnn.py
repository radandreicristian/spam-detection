from typing import Any, Optional

import torch
import torch.nn as nn
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW
from torchmetrics import F1, Recall
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pad_packed_sequence, pack_padded_sequence
from models.base_model import BaseSpamClassifier


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

        self.rnn = nn.RNN(input_size=self.d_input,
                          hidden_size=self.d_hidden,
                          num_layers=self.n_layers,
                          nonlinearity=self.activation,
                          dropout=self.p_dropout,
                          bidirectional=self.bidirectional,
                          batch_first=True)

        self.d_hidden_mlp = self.d_hidden * self.bidirectional_multiplier

        self.mlp = nn.ModuleList([
            nn.Linear(in_features=self.d_hidden_mlp,
                      out_features=self.d_hidden_mlp // 2),

            nn.Linear(in_features=self.d_hidden_mlp // 2,
                      out_features=1)
        ])

        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCELoss()

        # https://stackoverflow.com/a/54816498/10831784
        for p in self.parameters():
            p.register_hook(lambda grad: torch.clamp(grad, -self.clip_value, self.clip_value))

        self.f1 = F1(num_classes=1).to(self.device)
        self.recall = Recall(num_classes=1).to(self.device)

        self.metrics = {"f1": self.f1,
                        "recall": self.recall}

    def forward(self, x):
        # x (batch_size, seq_len, d_embedding)
        x = pad_sequence()

        # features (batch_size, seq_len, D * d_hidden), D=2 if bidirectional, 1 otherwise
        features, _ = self.rnn(x)

        # take only the output of the last cell
        # features (batch_size, 1, D * d_hidden)

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
        embeddings = batch['sequence']
        labels = batch['labels']
        predictions = self(embeddings)

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
        optimizer = AdamW(self.parameters(),
                          lr=self.learning_rate)
        return optimizer
