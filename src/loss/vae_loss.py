import torch
from torch import nn

from src.model.VAE.modules import DiagonalGaussianDistribution


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
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

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
                input=block_type_logits.permute(0, 4, 1, 2, 3),
                target=block_type_grid,
            )
        }


class KLLoss(nn.Module):
    def __init__(self, kl_weight=1.0):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, latents: DiagonalGaussianDistribution, **batch):
        return {"kl_loss": latents.kl().mean() * self.kl_weight}


class VAELoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, kl_weight=1.0):
        super().__init__()
        self.block_type_loss = BlockTypeLoss()
        self.attribute_loss = AttributeLoss()
        self.kl_loss = KLLoss(kl_weight)

    def forward(self, **batch):
        return_dict = dict()
        return_dict.update(self.block_type_loss(**batch))
        return_dict.update(self.attribute_loss(**batch))
        return_dict.update(self.kl_loss(**batch))

        total_loss = 0

        for key in return_dict:
            total_loss += return_dict[key]
        return_dict.update({"loss": total_loss})
        return return_dict
