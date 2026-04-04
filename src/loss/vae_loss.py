import torch
from torch import nn

from src.model.VAE.modules import DiagonalGaussianDistribution
from src.utils.io_utils import ROOT_PATH, read_json
from src.utils.model_utils import make_class_weights


class AttributeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        attributes_values: dict[str, torch.Tensor],
        attributes_logits: dict[str, torch.Tensor],
        **batch,
    ):
        """
        Loss function for blocks

        Args:
            attributes_data (List): a list of dicts of coords and dict of attributes and values tensor
            reconstructed_attributes_data (List): a list of dicts of coords and dict of attributes and values logits tensor
        Returns:
            loss (Tensor): calculated loss function
        """
        loss_dict = dict()
        for key in attributes_values:
            if len(attributes_logits[key]) and len(attributes_values[key]):
                loss_dict[f"{key}_loss"] = self.loss(
                    input=attributes_logits[key],
                    target=attributes_values[key],
                )
            else:
                loss_dict[f"{key}_loss"] = torch.tensor(
                    0.0, device=attributes_values[key].device
                )

        return loss_dict


class BlockTypeLoss(nn.Module):
    def __init__(self, block_data_path="./src/block_data"):
        super().__init__()
        block2idx = read_json(ROOT_PATH / block_data_path / "block2idx.json")
        statistics = read_json(ROOT_PATH / block_data_path / "statistics.json")

        weight = torch.zeros_like(len(block2idx))
        for k, v in statistics.items():
            weight[block2idx[k]] = v
        mask = weight > 0
        weight[mask] = make_class_weights(weight[mask])

        self.loss = nn.CrossEntropyLoss(weight)

    def forward(
        self,
        block_type_grid: torch.Tensor,
        block_type_logits: torch.Tensor,
        **batch,
    ) -> dict[str, torch.Tensor]:
        """
        Loss function for blocks

        Args:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L) representing ground-truth block types
            reconstructed_block_type_grid (Tensor): a tensor of shape (B, W, H, L, num_blocks) representing model output reconstruction
        Returns:
            loss (Tensor): calculated loss function
        """
        return {
            "block_type_loss": self.loss(
                input=block_type_logits.permute(0, 4, 1, 2, 3), target=block_type_grid
            )
        }


class KLLoss(nn.Module):
    def __init__(self, kl_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, latents: DiagonalGaussianDistribution, **batch):
        if self.kl_weight == 0:
            loss = torch.tensor(0, device=latents.parameters.device)
        else:
            loss = latents.kl().mean() * self.kl_weight
        return {"kl_loss": loss}


class FeatureLoss(nn.Module):
    def __init__(self, loss_type: str = "L1", feature_loss_weight: float = 1.0):
        super().__init__()
        self.weight = feature_loss_weight
        if loss_type is None:
            self.loss = None
        elif loss_type == "L1":
            self.loss = nn.L1Loss()
        elif loss_type == "L2":
            self.loss = nn.MSELoss()
        else:
            raise RuntimeError(f"Invalid Loss Type: {loss_type}")

    def forward(self, gt_features, pred_features, **batch):
        if self.loss is None:
            loss = torch.tensor(0, device=gt_features.device)
        else:
            loss = self.loss(pred_features, gt_features) * self.weight
        return {"feature_loss": loss}


class VAELoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, kl_weight=1.0, feature_loss_type=None, feature_loss_weight=1.0):
        super().__init__()
        self.block_type_loss = BlockTypeLoss()
        self.attribute_loss = AttributeLoss()
        self.kl_loss = KLLoss(kl_weight)
        self.feature_loss = FeatureLoss(feature_loss_type, feature_loss_weight)

    def forward(self, **batch):
        return_dict = dict()
        return_dict.update(self.block_type_loss(**batch))
        return_dict.update(self.attribute_loss(**batch))
        return_dict.update(self.kl_loss(**batch))
        return_dict.update(self.feature_loss(**batch))

        total_loss = 0

        for key in return_dict:
            total_loss += return_dict[key]
        return_dict.update({"loss": total_loss})
        return return_dict
