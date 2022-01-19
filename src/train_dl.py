import logging
from pathlib import Path

import hydra
import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data.utils import create_vocabulary, create_datasets
from models.embedders import EmbedderFactory
from models.model_factory import DlModelFactory
from src.data.datamodule import SpamDataModule

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@hydra.main(config_path='conf',
            config_name="dl_config.yaml")
def train(cfg: DictConfig):
    project_root_path = Path(__file__).parents[1].absolute()

    # Create datasets
    splits = ('train', 'valid', 'test')
    data_cfg = cfg.get("data", {})

    preprocessed_data_path = data_cfg.get("preprocessed_path")
    preprocessed_data_df = pd.read_csv(Path(project_root_path / preprocessed_data_path))

    paths = tuple([Path(project_root_path / f'data/processed/{split}/data.csv') for split in splits])
    _ = create_datasets(preprocessed_data_df, splits, paths)

    # Create the vocabulary
    content = preprocessed_data_df['content']
    vocab, word2idx = create_vocabulary(content, **data_cfg)

    model_cfg = cfg.get("model", {})

    embedder_cfg = cfg.get("embeddings", {})
    training_cfg = cfg.get("training", {})

    n_epochs = training_cfg.get("n_epochs")

    batch_size = training_cfg.get("batch_size")
    # Create the embedder
    embedder = EmbedderFactory.get_embedder(vocab=vocab,
                                            word2idx=word2idx,
                                            **embedder_cfg)

    # Create the model
    model = DlModelFactory.create_model(embedder=embedder,
                                        **model_cfg)

    # Create the data modules
    datamodule = SpamDataModule(paths=paths,
                                batch_size=batch_size,
                                word2idx=word2idx)

    # Create the callbacks
    checkpoint_callback = ModelCheckpoint(monitor="valid_loss",
                                          filename="spam_detector_{epoch:02d}_{valid_loss:.2f}",
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

    best_model_path = checkpoint_callback.best_model_path
    best_model = DlModelFactory.load_from_checkpoint(checkpoint_path=best_model_path,
                                                     embedder=embedder,
                                                     **model_cfg)

    trainer.test(model=best_model,
                 datamodule=datamodule)


if __name__ == '__main__':
    pl.seed_everything(21)
    train()
