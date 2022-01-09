from models.lstm import LstmSpamClassifier
from models.rnn import RnnSpamClassifier

import pytorch_lightning as pl


class ClassificationModelFactory:

    @staticmethod
    def get_model(model: str,
                  *args,
                  **kwargs) -> pl.LightningModule:
        if model == "rnn":
            return RnnSpamClassifier(**kwargs)
        elif model == "lstm":
            return LstmSpamClassifier(**kwargs)
        else:
            raise ValueError(f'"{model}" is not a valid classification model.')
