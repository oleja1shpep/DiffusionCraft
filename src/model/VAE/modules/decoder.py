import json
from pathlib import Path

import torch
from torch import nn

from src.utils.io_utils import ROOT_PATH, read_json
from src.utils.model_utils import get_head_key


class AttributeDecoder(nn.Module):
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
        self.attributes_defaults = read_json(
            ROOT_PATH / path_to_block_data / "attributes_defaults.json"
        )
        self.block_attributes_defaults = read_json(
            ROOT_PATH / path_to_block_data / "block_attributes_defaults.json"
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
            self.heads[key] = nn.Linear(self.D, len(values))

    def forward(
        self,
        block_type_grid: torch.Tensor,
        features: torch.Tensor,
        attributes_masks: dict[str, torch.Tensor],
        **batch,
    ) -> list[dict[str, dict[str, torch.Tensor]]]:
        """
        Decodes attributes

        Args:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L) consisting of block indexes.
            features (Tensor): a tensor of shape (B, W, H, L, D) consisting of block features
        Returns:
            attributes_data (List): a list of dicts containing attribute data for each coordinate
        """
        B, W, H, L = block_type_grid.shape

        # for each pair <attr, values> get logits of shape (N, len(values))
        attributes_data = dict()  # attr-pair : values

        for attr, values in self.non_default_attribute_pairs:
            head_key = get_head_key(attr, values)
            mask = attributes_masks[head_key]
            attr_logits = self.heads[head_key](features[mask])  # (N, len(values))
            attributes_data[head_key] = attr_logits

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
        self.num_blocks = len(
            read_json(ROOT_PATH / path_to_block_data / "idx2block.json")
        )

        self.head = nn.Linear(self.D, self.num_blocks)

    def forward(self, features: torch.Tensor):
        """
        Encodes block type

        Args:
            x (Tensor): a tensor of shape (B, W, H, L, D) consisting of features.
        Returns:
            block_type_grid (Tensor): a tensor of shape (B, W, H, L, num_blocks) representing the encoded block type grid.
        """

        return self.head(features)
