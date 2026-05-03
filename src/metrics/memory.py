import numpy as np
import torch

from src.metrics.base_metric import BaseMetric


class MaxMemoryAllocated(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.allocated_mem = []
        super().__init__(*args, **kwargs)

    def reset(self):
        self.allocated_mem = []

    def result(self):
        return {self.name: np.mean(self.allocated_mem)}

    def __call__(self, idxs, **batch):
        """
        Metric calculation logic.

        Args:
            idxs (Tensor): batch indexes.
        Returns:
            metric (float): calculated metric.
        """

        self.allocated_mem.append(torch.cuda.max_memory_allocated() / 1024**3)  # Gb
        torch.cuda.reset_peak_memory_stats()
        return self.allocated_mem[-1]
