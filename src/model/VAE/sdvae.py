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
        attn_layers=[],
        use_pred_masks=False,
        posterior_mode=None,
    ):
        """
        Args:
            channels (Int) : the dim of Embeddings.
            z_channels (Int) : the number of channels of latents.
            num_layers (Int) : layers of downsampling.
            num_res_blocks (Int) : number of ResnetBlocks in downsampling.
            attn_layers (List) : idxs of layers with Attention
            use_pred_masks (bool) : whether to calc masks on pred_block_grid
            posterior_mode (str) : 'sample' or 'mode'. Overrides sample_posterior arguement if forward
        """
        super().__init__()
        self.posterior_mode = posterior_mode

        self.encoder = Encoder(
            channels, num_layers, z_channels, num_res_blocks, attn_layers
        )
        self.decoder = Decoder(
            channels,
            num_layers,
            z_channels,
            num_res_blocks,
            attn_layers,
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

    def forward(self, sample_posterior=True, **batch):
        posterior, gt_features = self.encode(**batch)  # (B, 2 * z_dim, w, h, l)
        if self.posterior_mode is None:
            if sample_posterior:
                z = posterior.sample()
            else:
                z = posterior.mode()
        else:
            if self.posterior_mode == "sample":
                z = posterior.sample()
            elif self.posterior_mode == "mode":
                z = posterior.mode()
            else:
                raise RuntimeError(f"No such mode for posterior: {self.posterior_mode}")
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

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
