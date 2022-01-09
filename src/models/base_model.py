import pytorch_lightning as pl


class BaseSpamClassifier(pl.LightningModule):

    def __init__(self):
        super(BaseSpamClassifier, self).__init__()

    def reset_all_metrics(self):
        metrics = self.metrics.values()
        for metric in metrics:
            metric.reset()

    def update_all_metrics(self, predictions, labels):
        metrics = self.metrics.values()
        for metric in metrics:
            metric.update(preds=predictions,
                          target=labels)

    def log_metrics(self,
                    prefix: str) -> None:

        if prefix not in ["train", "valid"]:
            raise ValueError("Invalid preffix (shoud be train or valid)")
        metrics = {f"{prefix}_{k}": v for k, v in self.metrics.items()}
        self.log_dict(metrics)

    # Training-step specific methods
    def on_train_epoch_start(self) -> None:
        self.reset_all_metrics()

    def on_train_epoch_end(self) -> None:
        self.log_metrics(prefix="train")
        self.reset_all_metrics()

    # Validation-step specific methods
    def on_validation_epoch_start(self) -> None:
        self.reset_all_metrics()

    def on_validation_epoch_end(self) -> None:
        self.log_metrics(prefix="valid")
        self.reset_all_metrics()
