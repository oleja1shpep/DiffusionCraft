from collections import defaultdict

import numpy as np
import torch

from src.metrics.base_metric import BaseMetric
from src.utils.io_utils import ROOT_PATH, read_json
from src.utils.model_utils import AIR_BLOCK_IDX


class BlockTypeAccuracy(BaseMetric):
    def __init__(self, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        self.accuracies = []
        super().__init__(*args, **kwargs)

    def reset(self):
        self.accuracies = []

    def result(self):
        return {self.name: np.mean(self.accuracies)}

    def __call__(
        self, block_type_grid: torch.Tensor, pred_block_type_grid: torch.Tensor, **batch
    ):
        """
        Metric calculation logic.

        Args:
            pred_block_type_grid (Tensor): model output predictions.
            block_type_grid (Tensor): ground-truth block types.
        Returns:
            metric (float): calculated metric.
        """
        gt_non_air_mask = block_type_grid != AIR_BLOCK_IDX
        self.accuracies.append(
            (block_type_grid[gt_non_air_mask] == pred_block_type_grid[gt_non_air_mask])
            .to(torch.float32)
            .mean()
            .item()
        )


class MacroRecall(BaseMetric):
    def __init__(
        self,
        filter_rare_classes=False,
        block_data_path="./src/block_data",
        val_stats_file="statistics_val.json",
        *args,
        **kwargs
    ):
        """
        Macro Recall (Accuracy) among blocks

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        self.block2idx = read_json(ROOT_PATH / block_data_path / "block2idx.json")
        self.allowed_classes = list(range(len(self.block2idx)))

        if filter_rare_classes:
            self.val_stats = read_json(ROOT_PATH / block_data_path / val_stats_file)
            self.allowed_classes = [
                self.block2idx[block]
                for block in self.val_stats
                if self.val_stats[block] >= 30
            ]

        self.accuracy = defaultdict(list)
        self.old_accruracy = []
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.accuracy = defaultdict(list)
        self.old_accruracy = []

    def result(self):
        accuracies = []

        for v in self.accuracy.values():
            accuracies.append(np.mean(v))

        return {
            self.name: np.mean(accuracies) if len(accuracies) else np.nan,
            "MacroBlockTypeAccuracy": np.mean(self.old_accruracy)
            if len(self.old_accruracy)
            else np.nan,
        }

    def __call__(
        self,
        block_type_grid: torch.Tensor,
        pred_block_type_grid: torch.Tensor,
        block_type_logits: torch.Tensor,
        **batch
    ):
        """
        Metric calculation logic.

        Args:
            pred_block_type_grid (Tensor): model output predictions.
            block_type_grid (Tensor): ground-truth block types.
        Returns:
            metric (float): calculated metric.
        """
        B = block_type_logits.shape[0]

        results = []
        for b in range(B):
            per_class_acc = 0
            present_classes = 0

            target = block_type_grid[b]
            pred = pred_block_type_grid[b]

            for c in self.allowed_classes:
                class_mask = target == c
                class_count = class_mask.sum().item()
                if class_count != 0:
                    correct = (pred[class_mask] == c).sum().item()
                    self.accuracy[c].append(correct / class_count)
                    if c != AIR_BLOCK_IDX:
                        present_classes += 1
                        per_class_acc += correct / class_count
            if present_classes:
                results.append(per_class_acc / present_classes)
        if len(results):
            self.old_accruracy.append(np.mean(results))


class AttributeAccuracy(BaseMetric):
    def __init__(self, block_equality=True, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        super().__init__(*args, **kwargs)

        self.accuracies = defaultdict(list)
        self.old_accuracies = defaultdict(list)
        self.block_equality = block_equality
        self.reset()

    def reset(self):
        self.old_accuracies = defaultdict(list)
        self.accuracies = defaultdict(list)

    def result(self):
        accuracies = []

        for values in self.accuracies.values():
            accuracies.append(np.mean(values))

        accuracies = np.array(accuracies)

        result = {
            self.name + "Min": accuracies.min() if len(accuracies) else np.nan,
            self.name + "Max": accuracies.max() if len(accuracies) else np.nan,
            self.name + "Mean": accuracies.mean() if len(accuracies) else np.nan,
            self.name + "Median": np.median(accuracies) if len(accuracies) else np.nan,
        }
        key = (
            "RawAttributeAccuracy"
            if self.name == "GlobalAttrAccuracy"
            else "AttributeAccuracy"
        )
        for suffix in self.old_accuracies:
            result.update(
                {
                    key + suffix: np.mean(self.old_accuracies[suffix])
                    if len(self.old_accuracies[suffix])
                    else np.nan
                }
            )

        return result

    def unaugment_tensor(self, x: torch.Tensor, augmentations: dict):
        B = x.size(0)
        for i in range(B):
            k = (4 - augmentations["rotation"][i]) % 4
            x[i] = torch.rot90(x[i], k, dims=[0, 2]).contiguous()

        flip_mask1 = augmentations["flip"][:, 1]
        x[flip_mask1] = torch.flip(x[flip_mask1], dims=[2]).contiguous()

        flip_mask0 = augmentations["flip"][:, 0]
        x[flip_mask0] = torch.flip(x[flip_mask0], dims=[0]).contiguous()

        return x

    def __call__(
        self,
        block_type_grid: torch.Tensor,
        attributes_values: dict[str, torch.Tensor],
        attributes_masks: dict[str, torch.Tensor],
        pred_block_type_grid: torch.Tensor,
        attributes_logits: dict[str, torch.Tensor],
        augmentations: dict,
        **batch
    ):
        """
        Metric calculation logic.

        Args:
            attributes_logits (dict): model output predictions for attributes.
            attributes_values (dict): ground-truth labels for attributes.
        Returns:
            metric (tuple): statistics of calculated metric.
        """

        # if apply this mask on attr mask it will leave only valid connections between gt and pred attributes
        if self.block_equality:
            # remove augmentations
            block_equality_mask = self.unaugment_tensor(
                block_type_grid == pred_block_type_grid, augmentations
            )

        result = []
        for head_key in attributes_values:
            if self.block_equality:
                attr_mask = block_equality_mask[attributes_masks[head_key]]  # (N, )

                gt_attributes = attributes_values[head_key][attr_mask]
                pred_attributes = attributes_logits[head_key][attr_mask].argmax(-1)
            else:
                gt_attributes = attributes_values[head_key]
                pred_attributes = attributes_logits[head_key].argmax(-1)

            if len(gt_attributes) and len(pred_attributes):
                acc = (gt_attributes == pred_attributes).to(torch.float32).mean().item()
                self.accuracies[head_key].append(acc)
                result.append(acc)
        if len(result) == 0:
            result = [0]
        self.old_accuracies["Min"].append(np.min(result))
        self.old_accuracies["Max"].append(np.max(result))
        self.old_accuracies["Mean"].append(np.mean(result))
        self.old_accuracies["Median"].append(np.median(result))
