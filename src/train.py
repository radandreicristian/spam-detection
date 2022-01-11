import logging
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data.split_data import create_vocabulary, create_datasets
from models.embedders import EmbedderFactory
from models.model_factory import ClassificationModelFactory
from src.data.datamodule import SpamDataModule

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@hydra.main(config_path='conf',
            config_name="config.yaml")
def main(cfg: DictConfig):
    project_root_path = Path(__file__).parents[1].absolute()

    # Create datasets
    splits = ['train', 'valid', 'test']
    data_cfg = cfg.get("data", {})

    preprocessed_data_path = data_cfg.get("preprocessed_path")
    preprocessed_data_df = pd.read_csv(Path(project_root_path / preprocessed_data_path))

    paths = [Path(project_root_path / f'data/processed/{split}/data.csv') for split in splits]
    create_datasets(preprocessed_data_df, splits, *paths)

    # Create the vocabulary
    vocab, word2idx = create_vocabulary(preprocessed_data_df, **data_cfg)

    model_cfg = cfg.get("model", {})

    embedder_cfg = cfg.get("embeddings", {})
    training_cfg = cfg.get("training", {})

    n_epochs = training_cfg.get("n_epochs")

    # Create the embedder
    embedder = EmbedderFactory.get_embedder(vocab=vocab,
                                            word2idx=word2idx,
                                            **embedder_cfg)

    # Create the model
    model = ClassificationModelFactory.get_model(embedder=embedder,
                                                 **model_cfg)

    b = 0

    # Create the data modules
    datamodule = SpamDataModule(paths=list(paths),
                                train_batch_size=32,
                                word2idx=word2idx)

    # Create the callbacks
    checkpoint_callback = ModelCheckpoint(monitor="valid_f1",
                                          filename="spam_detector_model",
                                          save_top_k=1,
                                          mode="min")

    # Create the trainer
    trainer = Trainer(gpus=-1,
                      callbacks=[checkpoint_callback],
                      max_epochs=n_epochs,
                      deterministic=True)

    # Train the model
    trainer.fit(model=model,
                datamodule=datamodule)


if __name__ == '__main__':
    pl.seed_everything(21)
    main()
