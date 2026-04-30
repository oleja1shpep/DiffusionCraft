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


class AttributeEncoder(nn.Module):
    def __init__(self, emb_dim=256, block_data_path="src/block_data"):
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
        self.filtered_blocks = read_json(
            ROOT_PATH / block_data_path / "filtered_blocks.json"
        )
        self.idx2block = read_json(ROOT_PATH / block_data_path / "idx2block.json")
        self.attr_pair2idxs = read_json(
            ROOT_PATH / block_data_path / "attr_pair2idxs.json"
        )

        for key in self.attr_pair2idxs:
            self.attr_pair2idxs[key] = torch.tensor(
                self.attr_pair2idxs[key], dtype=torch.long
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


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv3d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool3d(x, kernel_size=2, stride=2)
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        channels=128,
        num_layers=3,
        z_channels=16,
        num_res_blocks=2,
        attn_layers=[],
        block_data_path="src/block_data",
    ):
        """
        The class for DownSampling Block Grid into latents

        Args:
            channels (Int) : the dim of Embeddings.
            z_channels (Int) : the number of channels of latents.
            num_layers (Int) : layers of downsampling.
            num_res_blocks (Int) : number of ResnetBlocks in downsampling.
            attn_layers (List) : idxs of layers with Attention
            block_data_path (str): path to the directory with block jsons.
        """
        super().__init__()
        self.block_type_encoder = BlockTypeEncoder(channels, block_data_path)
        self.attribute_encoder = AttributeEncoder(channels, block_data_path)

        self.num_layers = num_layers
        self.channels = channels
        self.z_channels = z_channels
        self.num_res_blocks = num_res_blocks

        self.down = nn.ModuleList()
        for i in range(self.num_layers):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = channels * (2**i)
            block_out = channels * (2 ** (i + 1))
            for _ in range(self.num_res_blocks):
                block.append(
                    ResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                    )
                )
                block_in = block_out
                if i in attn_layers:
                    attn.append(AttnBlock(block_in))

            down = nn.Module()
            down.block = block
            down.attn = attn
            down.downsample = Downsample(block_in)

            self.down.append(down)

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

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(
            block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, **batch):
        """
        Encodes attributes and block type

        Args:
            batch (dict): a dict representing batch of data samples.
        Returns:
            latents (Tensor): a tensor of shape (B, z_dim * 2, w, h, l)
        """
        features = self.block_type_encoder(**batch) + self.attribute_encoder(
            **batch
        )  # (B, W, H, L, D)

        h = features.permute(0, 4, 1, 2, 3)  # (B, D, W, H, L)

        # downsampling
        for i_level in range(self.num_layers):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                # if handle attn blocks
                if len(self.down[i_level].attn):
                    h = self.down[i_level].attn[i_block](h)
            h = self.down[i_level].downsample(h)

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, features.detach()  # detach for encoder endependence from decoder


class DCEncoder(Encoder):
    def space2channel(self, x: torch.Tensor, p: int = 2):
        """
        x: tensor of shape (B, D, W, H, L)
        """

        B, D, W, H, L = x.shape

        x = x.reshape(B, D, W // p, p, H // p, p, L // p, p)
        x = x.permute(0, 1, 2, 4, 6, 3, 5, 7)
        x = x.reshape(B, D, W // p, H // p, L // p, p**3)
        x = x.permute(0, 5, 1, 2, 3, 4)
        return x.reshape(B, D * p**3, W // p, H // p, L // p)

    def channel_avg(self, x: torch.Tensor, groups: int = 4):
        """
        x: tensor of shape (B, D, W, H, L)
        """
        B, D, W, H, L = x.shape
        return x.reshape(B, groups, D // groups, W, H, L).mean(1)

    def forward(self, **batch):
        """
        Encodes attributes and block type

        Args:
            batch (dict): a dict representing batch of data samples.
        Returns:
            latents (Tensor): a tensor of shape (B, z_dim * 2, w, h, l)
        """
        features = self.block_type_encoder(**batch) + self.attribute_encoder(
            **batch
        )  # (B, W, H, L, D)

        h = features.permute(0, 4, 1, 2, 3)  # (B, D, W, H, L)

        # downsampling
        for i_level in range(self.num_layers):
            # space-to-channel
            p = self.channel_avg(self.space2channel(h))
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                # if handle attn blocks
                if len(self.down[i_level].attn):
                    h = self.down[i_level].attn[i_block](h)
            h = self.down[i_level].downsample(h) + p

        # middle
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # end
        p = self.channel_avg(
            h, groups=self.channels * 2**self.num_layers // (2 * self.z_channels)
        )
        h = self.conv_out(h) + p
        return h, features.detach()  # detach for encoder endependence from decoder
