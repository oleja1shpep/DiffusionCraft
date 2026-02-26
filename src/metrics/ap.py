import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve

from src.metrics.base_metric import BaseMetric
from src.utils.model_utils import AIR_BLOCK_IDX


class AP(BaseMetric):
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
        self, block_type_grid: torch.Tensor, block_type_logits: torch.Tensor, **batch
    ):
        """
        Metric calculation logic.

        Args:
            block_type_grid (Tensor): ground-truth block types.
            block_type_logits (Tensor): model output predictions for block types.
        Returns:
            metric (Tensor): calculated metric.
        """
        B = len(block_type_grid)
        air_target = (
            (block_type_grid.detach() == AIR_BLOCK_IDX)
            .to(torch.int32)
            .reshape(B, -1)
            .cpu()
            .numpy()
            .astype(np.int32)
        )
        air_logits = (
            block_type_logits[..., AIR_BLOCK_IDX].reshape(B, -1).detach().cpu().numpy()
        )

        results = torch.zeros(B)
        for b in range(B):
            logits = air_logits[b]
            y_true = air_target[b]
            precision, recall, _ = precision_recall_curve(y_true, logits)

            results[b] = auc(recall, precision)

        return results.mean()
