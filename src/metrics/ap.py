import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve

from src.metrics.base_metric import BaseMetric
from src.utils.model_utils import AIR_BLOCK_IDX


class AP(BaseMetric):
    def __init__(self, air_only=True, *args, **kwargs):
        """
        Example of a nested metric class. Applies metric function
        object (for example, from TorchMetrics) on tensors.

        Notice that you can define your own metric calculation functions
        inside the '__call__' method.

        Args:
            metric (Callable): function to calculate metrics.
            device (str): device for the metric calculation (and tensors).
        """
        self.air_only = air_only
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
        num_classes = block_type_logits.shape[-1]
        B = len(block_type_grid)

        allowed_classes = list(range(num_classes))
        if self.air_only:
            allowed_classes = [AIR_BLOCK_IDX]
        else:
            allowed_classes.remove(AIR_BLOCK_IDX)

        results = []
        for b in range(B):
            mAP = 0
            present_classes = 0
            for c in allowed_classes:
                target = (
                    (block_type_grid[b] == c).flatten().cpu().numpy().astype(np.int32)
                )
                logits = (
                    block_type_logits[b, :, :, :, c].detach().flatten().cpu().numpy()
                )
                if target.sum() != 0:
                    precision, recall, _ = precision_recall_curve(target, logits)
                    mAP += auc(recall, precision)
                    present_classes += 1
            if present_classes:
                results.append(mAP / present_classes)

        if results:
            return torch.tensor(results).mean().item()
        else:
            return 0
