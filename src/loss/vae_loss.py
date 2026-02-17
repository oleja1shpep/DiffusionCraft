import torch
from torch import nn


class AttributeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        attributes_data: list[dict[str, dict[str, torch.Tensor]]],
        reconstructed_attributes_data: list[dict[str, dict[str, torch.Tensor]]],
        **batch
    ):
        """
        Loss function for blocks

        Args:
            attributes_data (List): a list of dicts of coords and dict of attributes and values tensor
            reconstructed_attributes_data (List): a list of dicts of coords and dict of attributes and values logits tensor
        Returns:
            loss (Tensor): calculated loss function
        """
        B = len(attributes_data)
        total_loss = 0
        # FIXME optimize and rework
        for b in range(B):
            for coords in attributes_data[b]:
                for attr in attributes_data[b][coords]:
                    total_loss += self.loss(
                        input=reconstructed_attributes_data[b][coords][attr],
                        target=attributes_data[b][coords][attr],
                    )

        return total_loss / B


class BlockTypeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(
        self,
        block_type_grid: torch.Tensor,
        reconstructed_block_type_grid: torch.Tensor,
        **batch
    ):
        """
        Loss function for blocks

        Args:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L) representing ground-truth block types
            reconstructed_block_type_grid (Tensor): a tensor of shape (B, W, H, L, num_blocks) representing model output reconstruction
        Returns:
            loss (Tensor): calculated loss function
        """
        return self.loss(
            input=reconstructed_block_type_grid.permute(0, 4, 1, 2, 3),
            target=block_type_grid,
        )


class VAELoss(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self):
        super().__init__()
        self.block_type_loss = BlockTypeLoss()
        self.attribute_loss = AttributeLoss()

    def forward(self, **batch):
        return {"loss": self.block_type_loss(**batch) + self.attribute_loss(**batch)}
