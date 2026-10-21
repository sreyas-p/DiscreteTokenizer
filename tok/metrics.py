"""Codebook health and reconstruction quality.

usage      = fraction of codes used at least once
perplexity = exp(-sum p log p) over code frequencies (= #codes if perfectly even, ~1 if collapsed)
dead codes = codes unused for the last `window` training steps
"""
import torch


class CodeStats:
    def __init__(self, num_codes, window=1000):
        self.num_codes, self.window = num_codes, window
        self.counts = torch.zeros(num_codes, dtype=torch.long)
        self.last_used = torch.full((num_codes,), -1, dtype=torch.long)

    def update(self, idx, step=None):
        idx = idx.detach().reshape(-1).cpu().long()
        self.counts += torch.bincount(idx, minlength=self.num_codes)
        if step is not None:
            self.last_used[idx.unique()] = step

    def reset_counts(self):
        self.counts.zero_()

    def usage(self):
        return (self.counts > 0).float().mean().item()

    def perplexity(self):
        p = self.counts.double() / self.counts.sum().clamp(min=1)
        return torch.exp(-(p * p.clamp(min=1e-12).log()).sum()).item()

    def dead(self, step):
        return int((step - self.last_used >= self.window).sum())


def psnr(x_hat, x):
    """Per-image PSNR on [0, 1], summed over the batch."""
    mse = (((x_hat + 1) / 2 - (x + 1) / 2) ** 2).flatten(1).mean(1)
    return (10 * torch.log10(1 / mse.clamp(min=1e-10))).sum().item()
