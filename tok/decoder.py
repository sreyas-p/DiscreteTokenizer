"""Frozen, pretrained decoder we borrow instead of training our own.

We use the VAE from Stable Diffusion, fine-tuned with an MSE-weighted loss:
    stabilityai/sd-vae-ft-mse  (MIT license)  https://huggingface.co/stabilityai/sd-vae-ft-mse

    Rombach, Blattmann, Lorenz, Esser, Ommer.
    "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022. arXiv:2112.10752

Its decoder maps a (B, 4, h, w) latent to a (B, 3, 8h, 8w) image, so a 64x64 image is an 8x8x4 latent.
None of its weights are trained here. Its encoder is used only as a teacher: its latent mean gives a
regression target for our encoder (see train.py), which makes training much faster.
"""
import torch
from torch import nn

MODEL_ID = "stabilityai/sd-vae-ft-mse"


class PretrainedDecoder(nn.Module):
    def __init__(self, vae=None, model_id=MODEL_ID):
        super().__init__()
        if vae is None:
            from diffusers import AutoencoderKL
            vae = AutoencoderKL.from_pretrained(model_id)
        self.vae = vae.eval().requires_grad_(False)
        self.scale = vae.config.scaling_factor  # 0.18215: puts latents at roughly unit variance
        self.latent_channels = vae.config.latent_channels

    def train(self, mode=True):  # always stays in eval mode
        return super().train(False)

    def decode(self, z_scaled):
        """z_scaled: (B, 4, h, w) latents in the scaled (~unit variance) space -> images in [-1, 1]."""
        return self.vae.decode(z_scaled / self.scale).sample.clamp(-1, 1)

    @torch.no_grad()
    def teacher_latents(self, x):
        """Scaled latent mean of the pretrained encoder: the target our encoder learns to imitate."""
        return self.vae.encode(x).latent_dist.mean * self.scale


def random_vae():
    """Same architecture as sd-vae-ft-mse, random weights. For tests and --smoke runs only."""
    from diffusers import AutoencoderKL

    with torch.random.fork_rng():
        torch.manual_seed(0)
        return AutoencoderKL(block_out_channels=[128, 256, 512, 512],
                             down_block_types=["DownEncoderBlock2D"] * 4,
                             up_block_types=["UpDecoderBlock2D"] * 4, layers_per_block=2, latent_channels=4,
                             norm_num_groups=32, sample_size=256, scaling_factor=0.18215)
