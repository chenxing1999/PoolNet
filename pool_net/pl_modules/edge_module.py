from .basic_module import BasePoolNetModule
from pool_net.models.poolnet import VggPoolNetEdge


class EdgePoolNetModule(BasePoolNetModule):
    def build_model(self):
        return VggPoolNetEdge()

    def forward(self, x, infer_edge=False):
        return self.core(x, infer_edge)

    def training_step(self, batch, batch_idx):
        x, mask, edge_mask = batch
        pred, pred_edge, pred_edges = self.core(x, infer_edge=True)


        loss = self.loss_function(pred, mask)
        edge_losses = []
        edge_losses.append(self.loss_function(pred_edge, edge_mask))

        for pred_edge in pred_edges:
            edge_losses.append(self.loss_function(pred_edge, edge_mask))

        loss = loss + sum(edge_losses)

        tensorboard_logs = {"train_loss": loss}

        return {
            "loss": loss,
            "log": tensorboard_logs,
        }

    def on_save_checkpoint(self, checkpoint):
        super(EdgePoolNetModule, self).on_save_checkpoint(checkpoint)
        checkpoint["use_edge"] = True
