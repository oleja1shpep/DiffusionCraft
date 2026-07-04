import torch
from torch import nn


class Flip(nn.Module):
    """
    Batch-version of Normalize for 1D Input.
    Used as an example of a batch transform.
    """

    def __init__(self, p=0.5):
        """
        Args:
            p (float): a probability of flip.
        """
        super().__init__()

        self.p = p

    def forward(self, items):
        """
        Args:
            items (dict): dict with tensor 'block_type_grid' of shape (W, H, L).
        Returns:
            items (dict): dict with flipped tensor.
        """
        block_type_grid = items["block_type_grid"]
        attributes_masks = items["attributes_masks"]

        if torch.rand(1, device=block_type_grid.device) < self.p:
            block_type_grid = torch.flip(block_type_grid, dims=[0])
            for head_key in attributes_masks:
                attributes_masks[head_key] = torch.flip(
                    attributes_masks[head_key], dims=[0]
                )

        if torch.rand(1, device=block_type_grid.device) < self.p:
            block_type_grid = torch.flip(block_type_grid, dims=[2])
            for head_key in attributes_masks:
                attributes_masks[head_key] = torch.flip(
                    attributes_masks[head_key], dims=[2]
                )

        items["block_type_grid"] = block_type_grid
        items["attributes_masks"] = attributes_masks
        return items
