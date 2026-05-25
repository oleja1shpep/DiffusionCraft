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

    def check_nan_inf(self, **batch):
        max_abs_param = 0
        for name, p in self.model.named_parameters():
            max_abs_param = max(max_abs_param, p.abs().max())
            p: torch.nn.Parameter
            if torch.any(p.isinf()):
                print(f"INF IN MODEL PARAMS: {name}")
            if torch.any(p.isnan()):
                print(f"NaN IN MODEL PARAMS: {name}")

            if p.grad is not None:
                if torch.any(p.grad.isinf()):
                    print(f"INF IN MODEL GRADS: {name}")
                if torch.any(p.grad.isnan()):
                    print(f"NaN IN MODEL GRADS: {name}")

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                if torch.any(batch[key].isinf()):
                    print(f"INF in batch element: {key}")

                if torch.any(batch[key].isnan()):
                    print(f"NaN in batch element: {key}")

            elif isinstance(batch[key], dict):
                for k in batch[key]:
                    if torch.any(batch[key][k].isinf()):
                        print(f"INF in batch element: {key} in key {k}")

                    if torch.any(batch[key][k].isnan()):
                        print(f"NaN in batch element: {key} in key {k}")

    def process_batch(self, step, epoch, batch: dict, tracker: MetricTracker):
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
        # batch = self.move_batch_to_device(batch) # do not need while using accelerate
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        current_step = (epoch - 1) * self.epoch_len + step

        parameter_stats = None
        opt_stats = None

        if self.is_train:
            with self.accelerator.accumulate(self.model):
                outputs = self.model(**batch, sample_posterior=self.is_train)
                batch.update(outputs)

                all_losses = self.criterion(**batch)
                batch.update(all_losses)

                loss = batch["loss"].detach().item()
                if self.config.trainer.get("debug", False) and current_step > 610:
                    print("loss:", loss)
                    if loss > 1e100:
                        self.logger.debug(
                            f"Step: {current_step} | HIGH LOSS: {loss} | Batch Indexes: {batch['idxs']}"
                        )
                        opt_stats = self.get_aggregated_optimizer_stats()
                        print(opt_stats)
                        import pdb

                        pdb.set_trace()

                self.accelerator.backward(batch["loss"])  # division on accum steps
                self._clip_grad_norm()
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()
                grad_norm = self._get_grad_norm()

                parameter_stats = self.get_parameter_stats()
                opt_stats = self.get_aggregated_optimizer_stats()

                if self.config.trainer.get("debug", False) and current_step > 610:
                    print("lr", self.lr_scheduler.get_last_lr()[0])
                    print("grad norm:", grad_norm)

                    print("Parameter stats:")
                    print("Weight:", *parameter_stats["weight"].items(), sep="\n\t")
                    print("Bias:", *parameter_stats["bias"].items(), sep="\n\t")

                    print("Latents-mu:")
                    print(f"\tmax:{batch['latents'].mean.max().item()}")
                    print(f"\tmin:{batch['latents'].mean.min().item()}")
                    print(f"\tmean:{batch['latents'].mean.mean().item()}")
                    print("Latents-logvar:")
                    print(f"\tmax:{batch['latents'].logvar.max().item()}")
                    print(f"\tmin:{batch['latents'].logvar.min().item()}")
                    print(f"\tmean:{batch['latents'].logvar.mean().item()}")

                    print("Logits:")
                    print(f"\tmax:{batch['block_type_logits'].max().item()}")
                    print(f"\tmin:{batch['block_type_logits'].min().item()}")
                    print(f"\tmean:{batch['block_type_logits'].mean().item()}")
                    print("Optimizer stats:", opt_stats)

                    if grad_norm > 7:
                        self.logger.debug(
                            f"Step: {current_step} | HIGH GRAD NORM: {grad_norm} | Batch Indexes: {batch['idxs']}"
                        )

                        import pdb

                        pdb.set_trace()
                tracker.update("grad_norm", grad_norm)
                self.optimizer.zero_grad()
        else:
            outputs = self.model(**batch, sample_posterior=self.is_train)
            batch.update(outputs)

            all_losses = self.criterion(**batch)
            batch.update(all_losses)

        if self.config.trainer.check_nan:
            self.check_nan_inf(**batch)
        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            tracker.update(loss_name, batch[loss_name].item())

        if tracker is not None:
            for met in tracker.metrics:
                value = met(
                    **batch, parameter_stats=parameter_stats, opt_stats=opt_stats
                )  # calculate metrics
                if met.name == "MaxMemoryAllocated":
                    if value > 55:
                        self.logger.debug(
                            f"Step: {current_step} | HIGH MEMORY CONSUMPTION: {round(value, 2)}Gb | Batch Indexes: {batch['idxs']}"
                        )

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
            self.log_structure_render(**batch, part=mode)
        else:
            self.log_structure_render(**batch, part=mode)

    def log_structure_render(
        self,
        block_type_grid: torch.Tensor,
        pred_block_type_grid: torch.Tensor,
        idxs,
        part,
        **batch,
    ):
        batch_size = block_type_grid.size(0) if part == "viz" else 1

        for b in range(batch_size):
            gt_render: Image.Image = render_block_grid(
                block_type_grid[b].detach().cpu().numpy(),
                self.block2color,
                self.idx2block,
            )
            pred_render = render_block_grid(
                pred_block_type_grid[b].detach().cpu().numpy(),
                self.block2color,
                self.idx2block,
            )
            if part == "viz":
                idx = idxs[b]
                self.writer.add_image(f"{idx}_gt_render", gt_render)
                self.writer.add_image(f"{idx}_pred_render", pred_render)
            else:
                self.writer.add_image("gt_render", gt_render)
                self.writer.add_image("pred_render", pred_render)

            gt_render.close()
            pred_render.close()
