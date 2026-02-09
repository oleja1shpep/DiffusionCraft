import json
from pathlib import Path

import torch
from torch import nn

from src.utils.io_utils import ROOT_PATH, read_json


class AttributeEncoder(nn.Module):
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
        self.filtered_blocks = read_json(path_to_block_data / "filtered_blocks.json")
        self.idx2block = read_json(path_to_block_data / "idx2block.json")

        self.heads = nn.ModuleDict()

        for pair in self.non_default_attribute_pairs:
            attr, values = pair
            key = f"{attr}:{sorted(values)}"
            self.heads[key] = nn.Embedding(
                num_embeddings=len(values), embedding_dim=self.D
            )

    def forward(
        self,
        block_type_grid: torch.Tensor,
        attributes_data: list[dict[str, dict[str, int]]],
        **batch,
    ):
        """
        Encodes attributes

        Args:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L) consisting of block indexes.
            attributes_data (List): a batch of dictionaries representing the attribute data of block grid.
        Returns:
            output (Tensor): a tensor of shape (B, W, H, L, D) representing the encoded attribute grid.
        """
        B, W, H, L = block_type_grid.shape
        output = torch.zeros(B, W, H, L, self.D)

        # need to optimize somehow but the structure of data is too complex
        for b in len(attributes_data):
            for coords in attributes_data:
                attr_dict = attributes_data[coords]
                x, y, z = list(map(int, coords.split("_")))
                block_idx = block_type_grid[b, x, y, z]
                block_type = self.idx2block[block_idx.item()]
                block_attr_dict = self.filtered_blocks[block_type]
                for attr in attr_dict:
                    assert (
                        attr in block_attr_dict
                    ), f"Attribute '{attr}' does not correspond to block '{block_type}'."

                    values = block_attr_dict[attr]
                    key = f"{attr}:{sorted(values)}"

                    assert (
                        key in self.heads
                    ), f"Attribute pair '{key}' must be present in heads' keys."

                    output[b, x, y, z] += self.heads[key](block_idx)

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
        self.num_blocks = len(read_json(path_to_block_data / "idx2block.json"))

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
    def __init__(self, emb_dim=256, path_to_block_data="src/block_data"):
        """
        The class for block grid encoder

        Args:
            emb_dim (int): the size of features dimension.
            path_to_block_data (str): path to the directory with block jsons.
        """
        super().__init__()
        self.block_type_encoder = BlockTypeEncoder(emb_dim, path_to_block_data)
        self.attribute_encoder = AttributeEncoder(emb_dim, path_to_block_data)

    def forward(self, **batch):
        """
        Encodes attributes and block type

        Args:
            batch (dict): a dict representing batch of data samples.
        Returns:
            output (Tensor): a tensor of shape (B, W, H, L, D) representing the encoded attribute grid.
        """

        return self.block_type_encoder(**batch) + self.attribute_encoder(**batch)
