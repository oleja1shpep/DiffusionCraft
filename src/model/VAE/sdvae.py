import torch
from torch import nn

from src.model.VAE.modules import Decoder, DiagonalGaussianDistribution, Encoder


class SDVAE(nn.Module):
    def __init__(
        self, channels=256, z_channels=16, n_layers=3, num_res_blocks=2, device="auto"
    ):
        """
        Args:
            channels (Int) : the dim of Embeddings.
            z_dim (Int) : the number of channels of latents.
            n_layers (Int) : layers of downsampling.
            num_res_blocks (Int) : number of ResnetBlocks in downsampling.
        """
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = Encoder(
            channels, n_layers, z_channels, num_res_blocks, device=device
        )
        self.decoder = Decoder(
            channels, n_layers, z_channels, num_res_blocks, device=device
        )

        self.quant_conv = nn.Conv3d(z_channels * 2, z_channels * 2, 1)
        self.post_quant_conv = nn.Conv3d(z_channels, z_channels, 1)

    def encode(self, **batch) -> DiagonalGaussianDistribution:
        h = self.encoder(**batch)  # (B, W, H, L, z_dim * 2)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(
        self, z: torch.Tensor, **batch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        z : tensor of latents of shape (B, W, H, L, z_dim)
        """
        z = self.post_quant_conv(z)  # (B, W, H, L, z_dim)
        return self.decoder(z, **batch)

    def forward(self, **batch):
        posterior = self.encode(**batch)  # (B, 2 * z_dim, w, h, l)
        z = posterior.mode()
        block_type_logits, pred_block_type_grid, attributes_logits = self.decode(
            z, **batch
        )

        return {
            "block_type_logits": block_type_logits,
            "attributes_logits": attributes_logits,
            "pred_block_type_grid": pred_block_type_grid,
            "latents": posterior,
        }
