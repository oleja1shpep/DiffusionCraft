import torch
from torch import nn

from src.model.VAE.modules import Decoder, DiagonalGaussianDistribution, Encoder


class SDVAE(nn.Module):
    def __init__(
        self,
        channels=192,
        z_channels=16,
        num_layers=3,
        num_res_blocks=2,
        use_pred_masks=False,
    ):
        """
        Args:
            channels (Int) : the dim of Embeddings.
            z_channels (Int) : the number of channels of latents.
            num_layers (Int) : layers of downsampling.
            num_res_blocks (Int) : number of ResnetBlocks in downsampling.
            use_pred_masks (bool) : whether to calc masks on pred_block_grid
        """
        super().__init__()

        self.encoder = Encoder(channels, num_layers, z_channels, num_res_blocks)
        self.decoder = Decoder(
            channels,
            num_layers,
            z_channels,
            num_res_blocks,
            use_pred_masks=use_pred_masks,
        )

        self.quant_conv = nn.Conv3d(z_channels * 2, z_channels * 2, 1)
        self.post_quant_conv = nn.Conv3d(z_channels, z_channels, 1)

    def post_init(self, device):
        for key in self.encoder.attribute_encoder.attr_pair2idxs:
            self.encoder.attribute_encoder.attr_pair2idxs[
                key
            ] = self.encoder.attribute_encoder.attr_pair2idxs[key].to(device)

        for key in self.decoder.attribute_decoder.attr_pair2idxs:
            self.decoder.attribute_decoder.attr_pair2idxs[
                key
            ] = self.decoder.attribute_decoder.attr_pair2idxs[key].to(device)

    def encode(self, **batch) -> DiagonalGaussianDistribution:
        h, features = self.encoder(**batch)  # (B, W, H, L, z_dim * 2)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior, features

    def decode(
        self, z: torch.Tensor, **batch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        z : tensor of latents of shape (B, W, H, L, z_dim)
        """
        z = self.post_quant_conv(z)  # (B, W, H, L, z_dim)
        return self.decoder(z, **batch)

    def forward(self, **batch):
        posterior, gt_features = self.encode(**batch)  # (B, 2 * z_dim, w, h, l)
        z = posterior.mode()
        (
            block_type_logits,
            pred_block_type_grid,
            attributes_logits,
            pred_attributes_masks,
            pred_features,
        ) = self.decode(z, **batch)

        return {
            "block_type_logits": block_type_logits,
            "attributes_logits": attributes_logits,
            "pred_attribures_masks": pred_attributes_masks,
            "pred_block_type_grid": pred_block_type_grid,
            "latents": posterior,
            "gt_features": gt_features,
            "pred_features": pred_features,
        }
