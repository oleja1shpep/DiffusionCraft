import torch
from torch import nn


class Rotate(nn.Module):
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

    def forward(self, items: dict):
        """
        Args:
            items (dict): dict with tensor 'block_type_grid' of shape (W, H, L).
        Returns:
            items (dict): dict with rotated tensor.
        """

        block_type_grid = items["block_type_grid"]
        attributes_masks = items["attributes_masks"]

        if torch.rand(1, device=block_type_grid.device) < self.p:
            k = torch.randint(0, 4, (1,), device=block_type_grid.device).item()
            block_type_grid = torch.rot90(block_type_grid, k, dims=[0, 2])
            for head_key in attributes_masks:
                attributes_masks[head_key] = torch.rot90(
                    attributes_masks[head_key], k, dims=[0, 2]
                )
        items["block_type_grid"] = block_type_grid
        items["attributes_masks"] = attributes_masks
        return items
