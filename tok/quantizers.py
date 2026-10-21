"""Four quantizers behind one interface: z (B, N, D) -> (z_q, indices (B, N), aux_loss).

A  plain VQ, loss-trained codebook (van den Oord et al. 2017, arXiv:1711.00937); written here
B  VQ + EMA + k-means init + dead-code restart     } vector-quantize-pytorch 1.31.6
C  low-dim (8) L2-normalized codes (ViT-VQGAN)     } github.com/lucidrains/vector-quantize-pytorch
D  FSQ, levels [8,5,5,5] (Mentzer et al. 2023)     }

All give about 10 bits per position (1024 codes, or 1000 for FSQ). They always run in fp32.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class VectorQuantizer(nn.Module):
    """Variant A."""

    def __init__(self, codebook_size=1024, dim=64, beta=0.25):
        super().__init__()
        self.beta = beta
        self.codebook = nn.Embedding(codebook_size, dim)
        self.codebook.weight.data.uniform_(-1 / codebook_size, 1 / codebook_size)

    def forward(self, z):
        E = self.codebook.weight
        flat = z.reshape(-1, z.shape[-1])
        dist = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ E.t() + E.pow(2).sum(1)[None]
        idx = dist.argmin(-1).view(z.shape[:-1])
        z_q = E[idx]
        loss = F.mse_loss(z_q, z.detach()) + self.beta * F.mse_loss(z, z_q.detach())
        return z + (z_q - z).detach(), idx, loss  # straight-through


class Quantizer(nn.Module):
    def __init__(self, variant="B", dim=64, codebook_size=1024, beta=0.25):
        super().__init__()
        from vector_quantize_pytorch import FSQ, VectorQuantize

        self.variant = variant
        if variant == "A":
            self.q, self.num_codes = VectorQuantizer(codebook_size, dim, beta), codebook_size
        elif variant == "B":
            self.q = VectorQuantize(dim=dim, codebook_size=codebook_size, decay=0.99, kmeans_init=True,
                                    kmeans_iters=10, threshold_ema_dead_code=2, commitment_weight=beta)
            self.num_codes = codebook_size
        elif variant == "C":
            self.q = VectorQuantize(dim=dim, codebook_size=codebook_size, codebook_dim=8,
                                    use_cosine_sim=True, commitment_weight=beta)
            self.num_codes = codebook_size
        elif variant == "D":
            levels = [8, 5, 5, 5]
            self.q, self.num_codes = FSQ(levels=levels, dim=dim), math.prod(levels)
        else:
            raise ValueError(f"variant must be A, B, C or D, got {variant!r}")

    @property
    def bits(self):
        return math.log2(self.num_codes)

    def forward(self, z):
        with torch.autocast(z.device.type, enabled=False):
            out = self.q(z.float())
        z_q, idx = out[0], out[1]
        aux = out[2].sum() if len(out) > 2 else z.new_zeros((), dtype=torch.float32)
        return z_q, idx, aux
