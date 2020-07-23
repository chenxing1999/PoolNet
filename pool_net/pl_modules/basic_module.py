import pytorch_lightning as pl
from pool_net.models.poolnet import VggPoolNet

import torch
from torch import nn
from torch.nn import functional as F

from sklearn import metrics
import numpy as np


class BasePoolNetModule(pl.LightningModule):
    def __init__(self, lr=5e-5, wd=0.0005, **kwargs):
        super(BasePoolNetModule, self).__init__()
        self.core = self.build_model()

        self.loss_ = nn.BCEWithLogitsLoss()

        # Optimizer parameters
        self.lr = lr
        self.wd = wd

    def build_model(self):
        return VggPoolNet()

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
        preds = self.core(x)

        loss = self.loss_function(preds, y)

        labels = y.cpu().long().numpy().flatten()
        labels = np.array(labels, dtype=np.uint8)

        preds = F.sigmoid(preds)
        preds = preds.cpu().numpy().flatten()

        return {"val_loss": loss, "labels": labels, "preds": preds}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()

        labels = [x["labels"] for x in outputs]
        preds = [x["preds"] for x in outputs]

        labels = np.concatenate(labels)
        preds = np.concatenate(preds)

        tensorboard_logs = {"val_loss": avg_loss}

        report = self.metric_report_gpu(preds, labels)


        tensorboard_logs.update(report)
        tqdm_dict = {"v-loss": avg_loss, "f-beta": report["f-beta"]}

        return {
            "val_loss": avg_loss,
            "log": tensorboard_logs,
            "progress_bar": tqdm_dict,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 15)
        return [optimizer], [scheduler]

    def metric_report(self, pred, label):
        """ Get list of metrics from predict and label
        Args:
            pred: 1-d array
            label: 1-d array
        Return:
            report (dict): contains:
                - precision
                - recall
                - f-beta
                - mae
        """
        prediction = pred
        prediction[pred >= 0.5] = 1
        prediction[pred < 0.5] = 0
        report = metrics.precision_recall_fscore_support(
            label, prediction, beta=0.3, average="binary"
        )
        # print(report)

        mae = np.mean(np.abs(pred - label))
        # print(mae)

        return {
            "precision": report[0],
            "recall": report[1],
            "f-beta": report[2],
            "mae": mae,
        }


    def metric_report_gpu(self, pred, label):
        # Convert pred and label to gpu
        pred = torch.tensor(pred).cuda()
        label = torch.tensor(label).long().cuda()

        prediction = (pred > 0.5).long()

        tp = (prediction * label).sum().to(torch.float32)
        tn = ((1-label) * (1-prediction)).sum().to(torch.float32)
        fp = ((1 - label) * prediction).sum().to(torch.float32)
        fn = (label * (1-prediction)).sum().to(torch.float32)

        eps = 1e-10

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)

        # Convert from gpu to cpu
        precision = precision.item()
        recall = recall.item()

        beta = 0.3
        f_beta = (1 + beta) * precision * recall / (beta * precision + recall)

        mae = torch.abs(pred - label).mean().item()
        return {
            "precision": precision,
            "recall": recall,
            "f-beta": f_beta,
            "mae": mae
        }
