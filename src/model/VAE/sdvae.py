import torch
from torch import nn

from src.model.VAE.modules import Decoder, Encoder


class SDVAE(nn.Module):
    def __init__(self, dim, device="auto"):
        super().__init__()

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.encoder = Encoder(dim, device=device)
        self.decoder = Decoder(dim, device=device)

    def forward(self, **batch):
        latents = self.encoder(**batch)  # (B, w, h, l, D)

        block_type_logits, attributes_logits = self.decoder(latents, **batch)

        return {
            "block_type_logits": block_type_logits,
            "attributes_logits": attributes_logits,
            "pred_block_type_grid": block_type_logits.argmax(-1),
        }
