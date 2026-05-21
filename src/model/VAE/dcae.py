import torch
from torch import nn

from src.model.VAE.modules import DCDecoder, DCEncoder, DiagonalGaussianDistribution


class DCAE(nn.Module):
    def __init__(
        self,
        channels=192,
        z_channels=16,
        num_layers=3,
        num_res_blocks=2,
        attn_layers=[],
        use_pred_masks=False,
    ):
        """
        Args:
            channels (Int) : the dim of Embeddings.
            z_channels (Int) : the number of channels of latents.
            num_layers (Int) : layers of downsampling.
            num_res_blocks (Int) : number of ResnetBlocks in downsampling.
            attn_layers (List) : idxs of layers with Attention
            use_pred_masks (bool) : whether to calc masks on pred_block_grid
        """
        super().__init__()

        self.encoder = DCEncoder(
            channels, num_layers, z_channels, num_res_blocks, attn_layers
        )
        self.decoder = DCDecoder(
            channels,
            num_layers,
            z_channels,
            num_res_blocks,
            attn_layers,
            use_pred_masks=use_pred_masks,
        )

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
        posterior = DiagonalGaussianDistribution(h)
        return posterior, features

    def decode(
        self, z: torch.Tensor, **batch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        z : tensor of latents of shape (B, W, H, L, z_dim)
        """
        return self.decoder(z, **batch)

    def forward(self, sample_posterior=True, **batch):
        posterior, gt_features = self.encode(**batch)  # (B, 2 * z_dim, w, h, l)
        if sample_posterior:
            z = posterior.sample()
        else:
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
