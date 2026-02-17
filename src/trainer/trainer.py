import numpy as np
import torch
from PIL import Image

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.model_utils import render_block_grid


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        with torch.amp.autocast(self.device, dtype=torch.float16):
            outputs = self.model(**batch)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

            if self.is_train:
                self.scaler.scale(
                    batch["loss"]
                ).backward()  # sum of all losses is always called loss
                self._clip_grad_norm()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_structure_render(**batch)
        else:
            self.log_structure_render(**batch)

    def log_structure_render(
        self, block_type_grid: torch.Tensor, pred_block_type_grid: torch.Tensor, **batch
    ):
        block_type_grid = block_type_grid[0].detach().cpu().numpy()
        pred_block_type_grid = pred_block_type_grid[0].detach().cpu().numpy()

        gt_render: Image.Image = render_block_grid(
            block_type_grid, self.block2color, self.idx2block
        )
        pred_render = render_block_grid(
            pred_block_type_grid, self.block2color, self.idx2block
        )

        self.writer.add_image("gt_render", gt_render)
        self.writer.add_image("pred_render", pred_render)

        gt_render.close()
        pred_render.close()
