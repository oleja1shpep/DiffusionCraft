import torch

from src.metrics.base_metric import BaseMetric
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
        super().__init__(*args, **kwargs)

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
        return (
            (block_type_grid[gt_non_air_mask] == pred_block_type_grid[gt_non_air_mask])
            .to(torch.float32)
            .mean()
            .item()
        )


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

        self.block_equality = block_equality

    def __call__(
        self,
        block_type_grid: torch.Tensor,
        attributes_values: dict[str, torch.Tensor],
        attributes_masks: dict[str, torch.Tensor],
        pred_block_type_grid: torch.Tensor,
        attributes_logits: dict[str, torch.Tensor],
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
            block_equality_mask = block_type_grid == pred_block_type_grid

        results = dict()

        for head_key in attributes_values:
            if self.block_equality:
                attr_mask = block_equality_mask[attributes_masks[head_key]]  # (N, )

                gt_attributes = attributes_values[head_key][attr_mask]
                pred_attributes = attributes_logits[head_key][attr_mask].argmax(-1)
            else:
                gt_attributes = attributes_values[head_key]
                pred_attributes = attributes_logits[head_key].argmax(-1)

            if len(gt_attributes) and len(pred_attributes):
                results[head_key] = (
                    (gt_attributes == pred_attributes).to(torch.float32).mean()
                )

        results = torch.tensor(list(results.values()))
        if len(results) == 0:
            results = torch.tensor([0.0])

        return {
            "Min": results.min().item(),
            "Max": results.max().item(),
            "Mean": results.mean().item(),
            "Median": results.median().item(),
        }
