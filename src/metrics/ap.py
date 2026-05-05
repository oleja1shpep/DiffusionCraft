from collections import defaultdict

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
        self.ap = defaultdict(list)
        self.old_ap = []
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.ap = defaultdict(list)
        self.old_ap = []

    def result(self):
        aps = []
        for values in self.ap.values():
            aps.append(np.mean(values))
        result = {self.name: np.mean(aps) if len(aps) else np.nan}

        if not self.air_only:
            result.update({"mAP": np.mean(self.old_ap) if len(self.old_ap) else np.nan})
        return result

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

        allowed_classes = [AIR_BLOCK_IDX] if self.air_only else list(range(num_classes))

        results = []
        for b in range(B):
            AP_old = 0
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
                    aucpr = auc(recall, precision)
                    self.ap[c].append(aucpr)
                    if c != AIR_BLOCK_IDX:
                        AP_old += aucpr
                        present_classes += 1
            if present_classes:
                results.append(AP_old / present_classes)

        if results:
            self.old_ap.append(np.mean(results))
