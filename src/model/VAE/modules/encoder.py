import json
from pathlib import Path

import torch
from torch import nn

from src.utils.io_utils import ROOT_PATH, read_json
from src.utils.model_utils import get_head_key


class AttributeEncoder(nn.Module):
    def __init__(self, emb_dim=256, path_to_block_data="src/block_data", device="cuda"):
        """
        The class for block attribute encoder

        Args:
            path_to_block_data (str): path to the directory with block jsons.
        """
        super().__init__()

        self.D = emb_dim

        self.non_default_attribute_pairs = read_json(
            ROOT_PATH / path_to_block_data / "non_default_attribute_pairs.json"
        )
        self.filtered_blocks = read_json(
            ROOT_PATH / path_to_block_data / "filtered_blocks.json"
        )
        self.idx2block = read_json(ROOT_PATH / path_to_block_data / "idx2block.json")
        self.attr_pair2idxs = read_json(
            ROOT_PATH / path_to_block_data / "attr_pair2idxs.json"
        )

        for key in self.attr_pair2idxs:
            self.attr_pair2idxs[key] = torch.tensor(
                self.attr_pair2idxs[key], device=device, dtype=torch.long
            )

        self.heads = nn.ModuleDict()

        for pair in self.non_default_attribute_pairs:
            attr, values = pair
            key = get_head_key(attr, values)
            self.heads[key] = nn.Embedding(
                num_embeddings=len(values), embedding_dim=self.D
            )

    def forward(
        self,
        block_type_grid: torch.Tensor,
        attributes_masks: dict[str, torch.Tensor],
        attributes_values: dict[str, torch.Tensor],
        **batch,
    ):
        """
        Encodes attributes

        Args:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L) consisting of block indexes.
            attributes_masks (dict): a dict with keys being attr-value pairs and values being batches of masks (B, W, H, L) for this pair.
            attributes_values (dict): a dict with keys being attr-value pairs and values being 1D tensors of values of corresponding attributes
        Returns:
            output (Tensor): a tensor of shape (B, W, H, L, D) representing the encoded attribute grid.
        """
        B, W, H, L = block_type_grid.shape
        output = torch.zeros(B, W, H, L, self.D, device=block_type_grid.device)

        # create masks
        for attr, values in self.non_default_attribute_pairs:
            head_key = get_head_key(attr, values)
            assert head_key in attributes_masks, f"Key '{head_key}' not in masks"
            assert head_key in attributes_values, f"Key '{head_key}' not in attr values"
            assert head_key in self.heads, f"Key '{head_key}' not in encoder heads"
            mask = attributes_masks[head_key]
            values = attributes_values[head_key]
            output[mask] += self.heads[head_key](values)

        return output


class BlockTypeEncoder(nn.Module):
    def __init__(self, emb_dim=256, path_to_block_data="src/block_data"):
        """
        The class for block type encoder

        Args:
            emb_dim (int): the size of features dimension.
            path_to_block_data (str): path to the directory with block jsons.
        """
        super().__init__()

        self.D = emb_dim
        self.num_blocks = len(
            read_json(ROOT_PATH / path_to_block_data / "idx2block.json")
        )

        self.head = nn.Embedding(self.num_blocks, self.D)

    def forward(self, block_type_grid: torch.Tensor, **batch):
        """
        Encodes block type

        Args:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L) consisting of block indexes.
        Returns:
            output (Tensor): a tensor of shape (B, W, H, L, D) representing the encoded block type grid.
        """

        return self.head(block_type_grid)


class GridEncoder(nn.Module):
    def __init__(self, emb_dim=256, path_to_block_data="src/block_data", device="cuda"):
        """
        The class for block grid encoder

        Args:
            emb_dim (int): the size of features dimension.
            path_to_block_data (str): path to the directory with block jsons.
        """
        super().__init__()
        self.block_type_encoder = BlockTypeEncoder(emb_dim, path_to_block_data)
        self.attribute_encoder = AttributeEncoder(emb_dim, path_to_block_data, device)

    def forward(self, **batch):
        """
        Encodes attributes and block type

        Args:
            batch (dict): a dict representing batch of data samples.
        Returns:
            output (Tensor): a tensor of shape (B, W, H, L, D) representing the encoded attribute grid.
        """

        return self.block_type_encoder(**batch) + self.attribute_encoder(**batch)
