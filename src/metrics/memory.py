import torch

from src.metrics.base_metric import BaseMetric


class MaxMemoryAllocated(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, idxs, **batch):
        """
        Metric calculation logic.

        Args:
            idxs (Tensor): batch indexes.
        Returns:
            metric (float): calculated metric.
        """

        value = torch.cuda.max_memory_allocated() / 1024**3  # Gb
        torch.cuda.reset_peak_memory_stats()
        return value
