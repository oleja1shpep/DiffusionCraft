import torch

from src.metrics.base_metric import BaseMetric


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
        return (block_type_grid == pred_block_type_grid).to(torch.float32).mean().item()


class AttributeAccuracy(BaseMetric):
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
        self,
        attributes_logits: dict[str, torch.Tensor],
        attributes_values: dict[str, torch.Tensor],
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
        results = []

        for key in attributes_values:
            if len(attributes_logits[key]) and len(attributes_values[key]):
                results.append(
                    (attributes_logits[key].argmax(-1) == attributes_values[key])
                    .to(torch.float32)
                    .mean()
                )
        results = torch.tensor(results)
        if len(results) == 0:
            return 0

        if self.name.endswith("Min"):
            return results.min().item()
        elif self.name.endswith("Max"):
            return results.max().item()
        elif self.name.endswith("Mean"):
            return results.mean().item()
        elif self.name.endswith("Median"):
            return results.median().item()
        else:
            raise RuntimeError
