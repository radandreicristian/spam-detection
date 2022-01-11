from models.lstm import LstmSpamClassifier
from models.rnn import RnnSpamClassifier

import pytorch_lightning as pl


class ClassificationModelFactory:

    @staticmethod
    def get_model(embedder,
                  *args,
                  **kwargs) -> pl.LightningModule:
        used_model = kwargs.get("used_model")
        used_model_config = kwargs.get(used_model)

        kwargs = {'embedder': embedder, **used_model_config}
        if used_model == "rnn":
            return RnnSpamClassifier(**kwargs)
        elif used_model == "lstm":
            return LstmSpamClassifier(**kwargs)
        else:
            raise ValueError(f'"{used_model}" is not a valid classification model.')
