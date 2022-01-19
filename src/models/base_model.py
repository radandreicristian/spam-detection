import pytorch_lightning as pl
from torchmetrics import F1


class BaseSpamClassifier(pl.LightningModule):

    def __init__(self):
        super(BaseSpamClassifier, self).__init__()

        self.train_f1 = F1(num_classes=1).to(self.device)
        self.valid_f1 = F1(num_classes=1).to(self.device)
        self.test_f1 = F1(num_classes=1).to(self.device)
