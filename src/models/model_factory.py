import sklearn.base
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from src.models.base_model import BaseSpamClassifier
from src.models.lstm import LstmSpamClassifier
from src.models.rnn import RnnSpamClassifier


class DlModelFactory:

    @staticmethod
    def create_model(embedder,
                     *args,
                     **kwargs) -> BaseSpamClassifier:
        used_model = kwargs.get("used_model")
        used_model_config = kwargs.get(used_model)

        kwargs = {'embedder': embedder, **used_model_config}
        if used_model == "rnn":
            return RnnSpamClassifier(**kwargs)
        elif used_model == "lstm":
            return LstmSpamClassifier(**kwargs)
        else:
            raise ValueError(f'"{used_model}" is not a valid classification model.')

    @staticmethod
    def load_from_checkpoint(checkpoint_path,
                             embedder,
                             *args,
                             **kwargs) -> BaseSpamClassifier:

        used_model = kwargs.get("used_model")
        used_model_config = kwargs.get(used_model)

        kwargs = {'embedder': embedder, **used_model_config}
        if used_model == "rnn":
            return RnnSpamClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                          **kwargs)
        if used_model == "lstm":
            return LstmSpamClassifier.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                           **kwargs)


class MlModelFactory:

    @staticmethod
    def create_model(*args,
                     **kwargs) -> sklearn.base.BaseEstimator:
        used_model = kwargs.get("used_model")
        model_config = kwargs.get(used_model)
        if used_model == "logistic":
            return LogisticRegression(**model_config)
        elif used_model == "svc":
            return SVC(**model_config)
        else:
            raise ValueError(f'"{used_model}" is not a valid classification model.')
