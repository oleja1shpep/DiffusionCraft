import json
from pathlib import Path

import torch
from torch import nn

from src.utils.io_utils import ROOT_PATH, read_json


class AttributeDecoder(nn.Module):
    def __init__(self, emb_dim=256, path_to_block_data="src/block_data"):
        """
        The class for block attribute encoder

        Args:
            path_to_block_data (str): path to the directory with block jsons.
        """
        super().__init__()

        self.D = emb_dim

        path_to_block_data = Path(path_to_block_data)

        self.non_default_attribute_pairs = read_json(
            path_to_block_data / "non_default_attribute_pairs.json"
        )
        self.attributes_defaults = read_json(
            path_to_block_data / "attributes_defaults.json"
        )
        self.block_attributes_defaults = read_json(
            path_to_block_data / "block_attributes_defaults.json"
        )
        self.filtered_blocks = read_json(path_to_block_data / "filtered_blocks.json")
        self.idx2block = read_json(path_to_block_data / "idx2block.json")

        self.heads = nn.ModuleDict()

        for pair in self.non_default_attribute_pairs:
            attr, values = pair
            key = f"{attr}:{sorted(values)}"
            self.heads[key] = nn.Linear(self.D, len(values))

    def forward(
        self,
        block_type_grid: torch.Tensor,
        x: torch.Tensor,
    ) -> list[dict[str, dict[str, torch.Tensor]]]:
        """
        Decodes attributes

        Args:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L) consisting of block indexes.
            x (Tensor): a tensor of shape (B, W, H, L, D) consisting of block features
        Returns:
            attributes_data (List): a list of dicts containing attribute data for each coordinate
        """
        B, W, H, L, _ = block_type_grid.shape

        attributes_data = [dict() for _ in range(B)]
        # need to optimize somehow but the structure of data is too complex
        for x in range(W):
            for y in range(H):
                for z in range(L):
                    key = f"{x}_{y}_{z}"
                    for b in range(B):
                        block_idx = block_type_grid[b, x, y, z]
                        block_type = self.idx2block[block_idx.item()]
                        block_attr_dict = self.filtered_blocks[block_type]
                        for attr in block_attr_dict:
                            if attr not in self.attributes_defaults:
                                if (
                                    attr
                                    not in self.block_attributes_defaults[block_type]
                                ):
                                    values = block_attr_dict[attr]
                                    head_key = f"{attr}:{sorted(values)}"
                                    logits = self.heads[head_key](x[b, x, y, z])
                                    if key not in attributes_data[b]:
                                        attributes_data[b][key] = dict()
                                    attributes_data[b][key][attr] = logits
        return attributes_data


class BlockTypeDecoder(nn.Module):
    def __init__(self, emb_dim=256, path_to_block_data="src/block_data"):
        """
        The class for block type encoder

        Args:
            emb_dim (int): the size of features dimension.
            path_to_block_data (str): path to the directory with block jsons.
        """
        super().__init__()

        self.D = emb_dim
        self.num_blocks = len(read_json(path_to_block_data / "idx2block.json"))

        self.head = nn.Linear(self.D, self.num_blocks)

    def forward(self, x: torch.Tensor):
        """
        Encodes block type

        Args:
            x (Tensor): a tensor of shape (B, W, H, L, D) consisting of features.
        Returns:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L, num_blocks) representing the encoded block type grid.
        """

        return self.head(x)
