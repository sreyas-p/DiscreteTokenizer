"""The tokenizer: our trainable encoder + quantizer, then the frozen pretrained decoder.

x (B,3,64,64) -> Encoder -> (B,64,8,8) -> 64 vectors -> Quantizer -> tokens (B,64)
  -> 1x1 conv to 4 channels -> frozen SD-VAE decoder -> x_hat (B,3,64,64)
"""
from torch import nn

from .quantizers import Quantizer


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(nn.ReLU(), nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(), nn.Conv2d(ch, ch, 1))

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    """64x64 -> 8x8 (three stride-2 convs)."""

    def __init__(self, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            ResBlock(256), ResBlock(256), nn.ReLU(),
            nn.Conv2d(256, latent_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


class Tokenizer(nn.Module):
    def __init__(self, decoder, variant="B", latent_dim=64, codebook_size=1024):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.quantizer = Quantizer(variant, latent_dim, codebook_size)
        self.to_decoder = nn.Conv2d(latent_dim, decoder.latent_channels, 1)  # trained
        self.decoder = decoder                                                # frozen

    def tokenize(self, x):
        z = self.encoder(x)
        B, D, h, w = z.shape
        z_q, idx, aux = self.quantizer(z.flatten(2).transpose(1, 2))
        return z_q.transpose(1, 2).reshape(B, D, h, w), idx, aux

    def forward(self, x):
        """Returns x_hat, the decoder-input latent (scaled space), token ids, quantizer loss."""
        z_q, idx, aux = self.tokenize(x)
        latent = self.to_decoder(z_q)
        return self.decoder.decode(latent), latent, idx, aux

    def trainable_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("decoder.")]

    def bits_per_image(self, image_size=64):
        return (image_size // 8) ** 2 * self.quantizer.bits
