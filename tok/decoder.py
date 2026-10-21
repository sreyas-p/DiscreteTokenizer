"""Frozen, pretrained VQ-VAE decoder we borrow instead of training our own.

We use the VQ-regularized autoencoder (VQ-f4, 8192-entry codebook, trained on CelebA-HQ faces) released with
Latent Diffusion:
    CompVis/ldm-celebahq-256, subfolder "vqvae"   https://huggingface.co/CompVis/ldm-celebahq-256

    Rombach, Blattmann, Lorenz, Esser, Ommer.
    "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR 2022. arXiv:2112.10752
    (autoencoder design from Esser, Rombach, Ommer. "Taming Transformers for High-Resolution Image
    Synthesis." CVPR 2021. arXiv:2012.09841)

Its decoder maps a (B, 3, h, w) latent to a (B, 3, 4h, 4w) image, so a 64x64 image is a 16x16x3 latent.
None of its weights are trained here. We feed it our own quantized latents and skip its codebook
(force_not_quantize=True). Its encoder is used only as a teacher: its pre-quantization latent gives a
regression target for our encoder (see train.py), which makes training much faster.
"""
import torch
from torch import nn

MODEL_ID = "CompVis/ldm-celebahq-256"
DOWNSAMPLE = 4


class PretrainedDecoder(nn.Module):
    def __init__(self, vqvae=None, model_id=MODEL_ID):
        super().__init__()
        if vqvae is None:
            from diffusers import VQModel
            vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae")
        self.vqvae = vqvae.eval().requires_grad_(False)
        self.latent_channels = vqvae.config.latent_channels
        self.downsample = DOWNSAMPLE

    def train(self, mode=True):  # always stays in eval mode
        return super().train(False)

    def decode(self, latent):
        """latent: (B, 3, h, w) -> images in [-1, 1]."""
        return self.vqvae.decode(latent, force_not_quantize=True).sample.clamp(-1, 1)

    @torch.no_grad()
    def teacher_latents(self, x):
        """Pre-quantization latent of the pretrained encoder: the target our encoder learns to imitate."""
        return self.vqvae.encode(x).latents


def random_vqvae():
    """Same architecture as the ldm-celebahq-256 VQ-f4 autoencoder, random weights. For tests and --smoke only."""
    from diffusers import VQModel

    with torch.random.fork_rng():
        torch.manual_seed(0)
        return VQModel(block_out_channels=[128, 256, 512], down_block_types=["DownEncoderBlock2D"] * 3,
                       up_block_types=["UpDecoderBlock2D"] * 3, layers_per_block=2, latent_channels=3,
                       num_vq_embeddings=8192, norm_num_groups=32, sample_size=256)
