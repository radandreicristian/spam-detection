import logging
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from data.utils import create_vocabulary, create_datasets
from src.models.model_factory import MlModelFactory

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@hydra.main(config_path='conf',
            config_name="ml_config.yaml")
def train(cfg: DictConfig):
    project_root_path = Path(__file__).parents[1].absolute()

    # Create datasets
    splits = ('train', 'valid', 'test')
    data_cfg = cfg.get("data", {})

    preprocessed_data_path = data_cfg.get("preprocessed_path")
    preprocessed_data_df = pd.read_csv(Path(project_root_path / preprocessed_data_path))

    paths = tuple([Path(project_root_path / f'data/processed/{split}/data.csv') for split in splits])
    datasets = create_datasets(preprocessed_data_df, splits, paths)

    # Create the vocabulary
    content = preprocessed_data_df['content']
    vocab, word2idx = create_vocabulary(content, **data_cfg)

    min_df = None if not data_cfg.get("ignore_rare_words", False) else data_cfg.get("rare_words_threshold", 0)

    train_df = datasets['train']
    valid_df = datasets['valid']
    test_df = datasets['test']

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                       ngram_range=(1, 2),
                                       lowercase=True,
                                       max_features=int(1e4),
                                       min_df=min_df)
    tfidf_vectorizer.fit(train_df['content'])

    model_cfg = cfg.get("model", {})

    # Model
    model = MlModelFactory.create_model(**model_cfg)
    train_features = tfidf_vectorizer.transform(train_df['content'])
    train_labels = train_df['label']

    model.fit(train_features, train_labels)

    test_features = tfidf_vectorizer.transform(test_df['content'])
    y_test_pred = model.predict(test_features)
    print(classification_report(y_test_pred, test_df['label']))


if __name__ == '__main__':
    pl.seed_everything(21)
    train()
