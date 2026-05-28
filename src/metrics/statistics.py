from collections import defaultdict

import numpy as np

from src.metrics.base_metric import BaseMetric
from src.model.VAE.modules import DiagonalGaussianDistribution


class ParameterStatistics(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.stats = defaultdict(list)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.stats = defaultdict(list)

    def result(self):
        result = dict()
        for key in self.stats:
            if "max" in key:
                result[f"{self.name}_{key}"] = np.max(self.stats[key])
            elif "min" in key:
                result[f"{self.name}_{key}"] = np.min(self.stats[key])
            else:
                result[f"{self.name}_{key}"] = np.mean(self.stats[key])
        return result

    def __call__(self, parameter_stats, **batch):
        """
        Metric calculation logic.

        Args:
            parameter_stats (Dict): dict with statistics
        Returns:
            metric (float): calculated metric.
        """

        for param_type in parameter_stats:
            for key in parameter_stats[param_type]:
                self.stats[f"{param_type}_{key}"].append(
                    parameter_stats[param_type][key]
                )


class OptimizerStatistics(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.stats = defaultdict(list)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.stats = defaultdict(list)

    def result(self):
        result = dict()
        for key in self.stats:
            if "max" in key:
                result[f"{self.name}_{key}"] = np.max(self.stats[key])
            elif "min" in key:
                result[f"{self.name}_{key}"] = np.min(self.stats[key])
            else:
                result[f"{self.name}_{key}"] = np.mean(self.stats[key])
        return result

    def __call__(self, opt_stats, **batch):
        """
        Metric calculation logic.

        Args:
            opt_stats (Dict): dict with statistics
        Returns:
            metric (float): calculated metric.
        """

        for stat_type in opt_stats:
            for key in opt_stats[stat_type]:
                self.stats[f"{stat_type}_{key}"].append(opt_stats[stat_type][key])


class LatentsStatistics(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.stats = defaultdict(list)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.stats = defaultdict(list)

    def result(self):
        result = dict()
        for key in self.stats:
            if "max" in key:
                result[f"{self.name}_{key}"] = np.max(self.stats[key])
            elif "min" in key:
                result[f"{self.name}_{key}"] = np.min(self.stats[key])
            else:
                result[f"{self.name}_{key}"] = np.mean(self.stats[key])
        return result

    def __call__(self, latents: DiagonalGaussianDistribution, **batch):
        """
        Metric calculation logic.

        Args:
            latents (DiagonalGaussianDistribution): latents
        Returns:
            metric (float): calculated metric.
        """

        mu = latents.mean.detach().cpu()
        logvar = latents.logvar.detach().cpu()

        self.stats["mu_min"].append(mu.min().item())
        self.stats["mu_max"].append(mu.max().item())
        self.stats["mu_mean"].append(mu.mean().item())

        self.stats["logvar_min"].append(logvar.min().item())
        self.stats["logvar_max"].append(logvar.max().item())
        self.stats["logvar_mean"].append(logvar.mean().item())


class BlockLogitsStatistics(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.stats = defaultdict(list)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.stats = defaultdict(list)

    def result(self):
        result = dict()
        for key in self.stats:
            if "max" in key:
                result[f"{self.name}_{key}"] = np.max(self.stats[key])
            elif "min" in key:
                result[f"{self.name}_{key}"] = np.min(self.stats[key])
            else:
                result[f"{self.name}_{key}"] = np.mean(self.stats[key])
        return result

    def __call__(self, block_type_logits, **batch):
        """
        Metric calculation logic.

        Args:
            block_type_logits (Tensor): logits of blocks
        Returns:
            metric (float): calculated metric.
        """

        self.stats["min"].append(block_type_logits.min().item())
        self.stats["max"].append(block_type_logits.max().item())
        self.stats["mean"].append(block_type_logits.mean().item())


class ActivationStatisitcs(BaseMetric):
    def __init__(self, *args, **kwargs):
        self.stats = defaultdict(list)
        super().__init__(*args, **kwargs)

    def reset(self):
        self.stats = defaultdict(list)

    def result(self):
        result = dict()
        for key in self.stats:
            if "max" in key:
                result[f"{self.name}_{key}"] = np.max(self.stats[key])
            elif "min" in key:
                result[f"{self.name}_{key}"] = np.min(self.stats[key])
            else:
                result[f"{self.name}_{key}"] = np.mean(self.stats[key])
        return result

    def __call__(self, layer_stats, **batch):
        """
        Metric calculation logic.

        Args:
            layer_stats (Dict): dict with statistics
        Returns:
            metric (float): calculated metric.
        """

        for layer_name in layer_stats:
            for key in layer_stats[layer_name]:
                # change if need to log for every layer separately
                self.stats[f"{key}"].append(layer_stats[layer_name][key])
