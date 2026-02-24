import torch
from torch import nn

from src.model.VAE.modules.layers import (
    AttnBlock,
    Normalize,
    ResnetBlock3D,
    nonlinearity,
)
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
        features: torch.Tensor,
        pred_block_type_grid: torch.Tensor,
        attributes_masks: dict[str, torch.Tensor],
        **batch,
    ) -> list[dict[str, dict[str, torch.Tensor]]]:
        """
        Decodes attributes

        Args:
            features (Tensor): a tensor of shape (B, W, H, L, D) consisting of block features.
            pred_block_type_grid (Tensor) : a tensor of shape (B, W, H, L) consisting of block type ids.
            attributes_masks (Dict): a dict of masks for each attribute_pair.
        Returns:
            attributes_data (List): a list of dicts containing attribute data for each coordinate.
        """

        # for each pair <attr, values> get logits of shape (N, len(values))
        attributes_data = dict()  # attr-pair : values

        for attr, values in self.non_default_attribute_pairs:
            head_key = get_head_key(attr, values)
            # if train mode get gt masks
            mask = attributes_masks[head_key]
            # if eval mode calculate new masks
            # else:
            #     mask = torch.isin(pred_block_type_grid, self.attr_pair2idxs[head_key])
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


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self,
        channels=128,
        n_layers=3,
        z_channels=16,
        num_res_blocks=2,
        block_data_path="src/block_data",
        device="cuda",
    ):
        """
        The class for DownSampling Block Grid into latents

        Args:
            emb_dim (int): the size of features dimension.
            block_data_path (str): path to the directory with block jsons.
        """
        super().__init__()
        self.n_layers = n_layers
        self.z_channels = z_channels
        self.num_res_blocks = num_res_blocks

        block_in = channels * (2**n_layers)

        # z to block_in
        self.conv_in = torch.nn.Conv3d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i in reversed(range(n_layers)):
            block = nn.ModuleList()
            block_out = channels * (2**i)
            for _ in range(num_res_blocks + 1):
                block.append(
                    ResnetBlock3D(in_channels=block_in, out_channels=block_out)
                )
                block_in = block_out

            up = nn.Module()
            up.block = block
            up.upsample = Upsample(block_in)
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.blocks_end = nn.Sequential(
            *[
                ResnetBlock3D(
                    in_channels=channels, out_channels=channels  # block_in = channels
                )
                for _ in range(num_res_blocks + 1)
            ]
        )
        self.norm_out = Normalize(channels)

        self.block_type_decoder = BlockTypeDecoder(channels, block_data_path)
        self.attribute_decoder = AttributeDecoder(channels, block_data_path, device)

        self.upsample_block = nn.Identity()

    def forward(self, z: torch.Tensor, **batch):
        """
        Decodes attributes and block type

        Args:
            z (Tensor) : a tensor of shape (B, z_dim, w, h, l)
            batch (Dict): a dict representing batch of data samples.
        Returns:
            output (Tuple[Tensor, dict]): A pair with first element being a tensor of shape (B, W, H, L, num_blocks) representing the block_type_logits and second element representing attribute data
        """

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)  # (B, block_in, w, h, l)

        # upsampling
        for i in reversed(range(self.n_layers)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i].block[i_block](h)
            h = self.up[i].upsample(h)

        h = self.norm_out(h)
        h = nonlinearity(h)  # (B, D, W, H, L)

        h = h.permute(0, 2, 3, 4, 1)  # (B, W, H, L, D)

        block_type_logits = self.block_type_decoder(h)  # (B, W, H, L, num_blocks)
        pred_block_type_grid = block_type_logits.argmax(-1)
        attributes_logits = self.attribute_decoder(h, pred_block_type_grid, **batch)
        return block_type_logits, pred_block_type_grid, attributes_logits
