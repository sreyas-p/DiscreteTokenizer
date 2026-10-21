"""Evaluate a checkpoint on the first 10,000 CelebA test images.

    python evaluate.py --ckpt runs/B/last.pt

Writes (or replaces) the variant's row in results/results.csv and saves results/recon_<variant>.png
(top row: originals, bottom row: reconstructions).
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from tok.data import RandomImages, celeba
from tok.decoder import PretrainedDecoder, random_vae
from tok.metrics import CodeStats, psnr
from tok.model import Tokenizer

COLUMNS = ["variant", "steps", "bits_per_image", "usage", "perplexity", "psnr", "ssim", "lpips"]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--eval-size", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--results", default="results")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    from torchmetrics.functional.image import structural_similarity_index_measure as ssim

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device)
    decoder = PretrainedDecoder(random_vae() if ck["smoke"] else None)
    model = Tokenizer(decoder, ck["variant"]).to(device).eval()
    model.load_state_dict(ck["model"], strict=False)  # decoder weights come from the hub, not the checkpoint

    data = RandomImages(args.eval_size, seed=1) if ck["smoke"] else celeba(args.data_root, "test", eval_size=args.eval_size)
    lpips_fn = None
    if not args.no_lpips:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval()

    stats = CodeStats(model.quantizer.num_codes)
    sums, n = {"psnr": 0.0, "ssim": 0.0, "lpips": 0.0}, 0
    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    for x in DataLoader(data, batch_size=args.batch_size):
        x = x.to(device)
        x_hat, _, idx, _ = model(x)
        stats.update(idx)
        sums["psnr"] += psnr(x_hat, x)
        sums["ssim"] += ssim((x_hat + 1) / 2, (x + 1) / 2, data_range=1.0, reduction="sum").item()
        if lpips_fn is not None:
            sums["lpips"] += lpips_fn(x_hat, x).sum().item()
        if n == 0:
            save_image((torch.cat([x[:8], x_hat[:8]]) + 1) / 2, out / f"recon_{ck['variant']}.png", nrow=8)
        n += x.shape[0]

    row = {"variant": ck["variant"], "steps": ck["step"], "bits_per_image": f"{model.bits_per_image():.1f}",
           "usage": f"{stats.usage():.4f}", "perplexity": f"{stats.perplexity():.1f}",
           "psnr": f"{sums['psnr'] / n:.3f}", "ssim": f"{sums['ssim'] / n:.4f}",
           "lpips": f"{sums['lpips'] / n:.4f}" if lpips_fn is not None else ""}
    path = out / "results.csv"
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r["variant"] != row["variant"]]
    rows = sorted(rows + [row], key=lambda r: r["variant"])
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(row)


if __name__ == "__main__":
    main()
