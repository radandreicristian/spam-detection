import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from models.embedders import EmbedderFactory

import pytorch_lightning as pl

from data.datamodule import SpamDataModule
from models.model_factory import ClassificationModelFactory

from pathlib import Path


@hydra.main(config_name="config.yaml")
def main(cfg: DictConfig):
    project_root_path = Path(__file__).parents[1].absolute()

    train_path = Path(project_root_path / 'data/processed/train/data.csv')
    valid_path = Path(project_root_path / 'data/processed/valid/data.csv')
    test_path = Path(project_root_path / 'data/processed/test/data.csv')

    model_name = cfg.get("model_name", "")
    model_cfg = cfg.get("model", {}).get(model_name, {})

    embedder_cfg = cfg.get("embeddings", {})
    training_cfg = cfg.get("training", {})

    n_epochs = training_cfg.get("n_epochs")
    max_seq_len = model_cfg.get("max_seq_len")

    # Create the embedder
    embedder = EmbedderFactory.get_embedder(**embedder_cfg, max_seq_len=max_seq_len)

    # Create the model
    model = ClassificationModelFactory.get_model(model_name, **model_cfg)

    # Create the data modules
    rnn_datamodule = SpamDataModule(train_path=train_path,
                                    valid_path=valid_path,
                                    test_path=test_path,
                                    train_batch_size=4,
                                    embedder=embedder)

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
                datamodule=rnn_datamodule)


if __name__ == '__main__':
    pl.seed_everything(21)
    main()
