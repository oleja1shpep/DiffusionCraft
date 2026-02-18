import torch
from torch import nn

from src.utils.io_utils import ROOT_PATH, read_json
from src.utils.model_utils import get_head_key


class AttributeDecoder(nn.Module):
    def __init__(self, emb_dim=256, block_data_path="src/block_data", device="cuda"):
        """
        The class for block attribute encoder

        Args:
            block_data_path (str): path to the directory with block jsons.
        """
        super().__init__()

        self.D = emb_dim

        self.non_default_attribute_pairs = read_json(
            ROOT_PATH / block_data_path / "non_default_attribute_pairs.json"
        )
        self.attributes_defaults = read_json(
            ROOT_PATH / block_data_path / "attributes_defaults.json"
        )
        self.block_attributes_defaults = read_json(
            ROOT_PATH / block_data_path / "block_attributes_defaults.json"
        )
        self.filtered_blocks = read_json(
            ROOT_PATH / block_data_path / "filtered_blocks.json"
        )
        self.idx2block = read_json(ROOT_PATH / block_data_path / "idx2block.json")
        self.attr_pair2idxs = read_json(
            ROOT_PATH / block_data_path / "attr_pair2idxs.json"
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
    def __init__(self, emb_dim=256, block_data_path="src/block_data"):
        """
        The class for block type encoder

        Args:
            emb_dim (int): the size of features dimension.
            block_data_path (str): path to the directory with block jsons.
        """
        super().__init__()

        self.D = emb_dim
        self.num_blocks = len(read_json(ROOT_PATH / block_data_path / "idx2block.json"))

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


class Decoder(nn.Module):
    def __init__(self, emb_dim=256, block_data_path="src/block_data", device="cuda"):
        """
        The class for DownSampling Block Grid into latents

        Args:
            emb_dim (int): the size of features dimension.
            block_data_path (str): path to the directory with block jsons.
        """
        super().__init__()
        self.block_type_decoder = BlockTypeDecoder(emb_dim, block_data_path)
        self.attribute_decoder = AttributeDecoder(emb_dim, block_data_path, device)

        self.upsample_block = nn.Identity()

    def forward(self, latents, **batch):
        """
        Decodes attributes and block type

        Args:
            latents (Tensor) : a tensor of shape (B, w, h, l, C)
            batch (Dict): a dict representing batch of data samples.
        Returns:
            output (Tuple[Tensor, dict]): A pair with first element being a tensor of shape (B, W, H, L, num_blocks) representing the block_type_logits and second element representing attribute data
        """
        features = self.upsample_block(latents)  # (B, W, H, L, D)
        block_type_logits = self.block_type_decoder(
            features
        )  # (B, W, H, L, num_blocks)
        attributes_logits = self.attribute_decoder(**batch, features=features)
        return block_type_logits, attributes_logits
