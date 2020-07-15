import pytorch_lightning as pl
from pool_net.models.poolnet import VggPoolNet

import torch
from torch import nn

class BasePoolNetModule(pl.LightningModule):
    def __init__(self, lr=5e-5, wd=0.0005, **kwargs):
        super(BasePoolNetModule, self).__init__()
        self.core = VggPoolNet()

        self.loss_ = nn.BCEWithLogitsLoss()

        # Optimizer parameters
        self.lr = lr
        self.wd = wd
    

    def forward(self, x):
        """
        Args:
            x (tensor B x 3 x W x H)
        Return:
            mask (tensor B x 1 x W x H)
        """
        return self.core(x)

    def loss_function(self, pred, label):
        return self.loss_(pred, label)
    

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self.core(x)

        loss = self.loss_function(pred, y)
        tensorboard_logs = {"train_loss": loss}

        return {
            "loss": loss,
            "log": tensorboard_logs,
        }

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.core(x)

        loss = self.loss_function(pred, y)
        tensorboard_logs = {"train_loss": loss}

        return {
            "loss": loss,
            "log": tensorboard_logs,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
        return [optimizer], [scheduler]

