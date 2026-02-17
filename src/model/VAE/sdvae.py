import torch
from torch import nn

from src.model.VAE.modules import AttributeDecoder, BlockTypeDecoder, GridEncoder


class SDVAE(nn.Module):
    def __init__(self, dim, device="auto"):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = GridEncoder(dim, device=device)
        self.block_decoder = BlockTypeDecoder(dim)

        # self.downsample = nn.Sequential(
        #     nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.Conv3d(dim, dim, kernel_size=3, stride=1, padding=1),
        # )

        # self.upsample = nn.Sequential(
        #     nn.ConvTranspose3d(dim, dim, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(),
        #     nn.ConvTranspose3d(dim, dim, kernel_size=3, stride=1, padding=1),
        # )

        self.attr_decoder = AttributeDecoder(dim, device=device)

    def forward(self, **batch):
        features = self.encoder(**batch)  # (B, W, H, L, D)
        _, W, H, L, _ = features.shape

        # features = features.permute(0, 4, 1, 2, 3) # (B, D, W, H, L)
        # latents = self.downsample(features)

        # features_reconstructed = self.upsample(latents).permute(0, 2, 3, 4, 1) # (B, W', H', L', D)
        features_reconstructed = features[:, :W, :H, :L]

        reconstructed_block_type_grid = self.block_decoder(features_reconstructed)
        reconstructed_attributes_values = self.attr_decoder(
            **batch, features=features_reconstructed
        )
        return {
            "reconstructed_block_type_grid": reconstructed_block_type_grid,
            "reconstructed_attributes_values": reconstructed_attributes_values,
        }
