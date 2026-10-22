"""Live demo: watch codebook collapse happen, on a laptop CPU, in about a minute.

    python demo.py                 # no downloads: random images + random-weight copy of the decoder
    python demo.py --real          # pretrained VQ-VAE decoder + CelebA test images from data/celeba

1. Tokenize: one image becomes a 16x16 grid of discrete token ids (~2,560 bits).
2. Collapse race: train the encoder + each quantizer (A, B, C, D) side by side and watch codebook usage.
   For speed, the frozen VQ-VAE's latents are computed once and the encoder is trained to match them
   (the latent-matching term of train.py's loss). The 33M-parameter decoder runs only at the end.
3. Saves demo_out/recon.png: originals (top row), then one row per quantizer.
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from tok.data import RandomImages, celeba
from tok.decoder import PretrainedDecoder, random_vqvae
from tok.metrics import CodeStats
from tok.model import Tokenizer

NAMES = {"A": "plain VQ (no fix)", "B": "EMA + k-means + restarts", "C": "low-dim cosine codes", "D": "FSQ"}


def bar(frac, width=30):
    n = round(frac * width)
    return "█" * n + "·" * (width - n)


def sparkline(counts, buckets=40):
    """Sorted code frequencies, most used first: shows whether a few codes take everything."""
    c = counts.sort(descending=True).values.float()
    c = c[: (len(c) // buckets) * buckets].view(buckets, -1).sum(1)
    c = c / c.max().clamp(min=1)
    return "".join(" ▁▂▃▄▅▆▇█"[int(v * 8)] for v in c)


def section(title):
    print(f"\n\033[1m{title}\033[0m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="pretrained decoder + CelebA (needs downloads/data)")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--variants", default="AB", help="e.g. ABCD for all four (slower)")
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--images", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", default="demo_out")
    args = ap.parse_args()
    torch.manual_seed(0)

    decoder = PretrainedDecoder(None if args.real else random_vqvae())
    data = celeba(args.data_root, "test", eval_size=args.images) if args.real else RandomImages(args.images, seed=1)
    x = torch.stack([data[i] for i in range(len(data))])
    with torch.no_grad():
        target = torch.cat([decoder.teacher_latents(b) for b in x.split(16)])

    section("1. An image becomes tokens")
    model = Tokenizer(decoder, "B")
    with torch.no_grad():
        _, ids, _ = model.tokenize(x[:1])
    grid = ids.view(16, 16)
    print(f"   one 64x64 image -> {grid.numel()} tokens x {model.quantizer.bits:.0f} bits = "
          f"{model.bits_per_image():.0f} bits (raw pixels: {64 * 64 * 3 * 8:,} bits)")
    for row in grid[:4]:
        print("   " + " ".join(f"{int(t):4d}" for t in row[:12]) + "  ...")
    print("   ...")

    section(f"2. Collapse race: {args.steps} training steps per quantizer, same encoder/data/seed/bits")
    results = {}
    for v in args.variants:
        torch.manual_seed(0)
        model = Tokenizer(decoder, v)
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=2e-4 * 5)  # higher lr: a short demo run
        stats = CodeStats(model.quantizer.num_codes)
        t0 = time.time()
        for step in range(1, args.steps + 1):
            i = torch.randint(0, len(x), (args.batch_size,))
            z_q, idx, q_loss = model.tokenize(x[i])
            loss = F.mse_loss(model.to_decoder(z_q), target[i]) + q_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 10 == 0 or step == args.steps:
                stats.reset_counts()
                with torch.no_grad():
                    _, idx_all, _ = model.tokenize(x)
                stats.update(idx_all)
                sys.stdout.write(f"\r   {v}  {NAMES[v]:<26} step {step:4d}  usage {bar(stats.usage())} "
                                 f"{stats.usage() * 100:5.1f}%")
                sys.stdout.flush()
        print(f"  ({time.time() - t0:.0f}s)")
        results[v] = (model, stats.counts.clone(), stats.usage(), stats.perplexity())

    section("3. Results  (usage = share of codes used, perplexity = effective number of codes)")
    print(f"   {'':3}{'quantizer':<28}{'usage':>8}{'perplexity':>12}   code frequencies, most used first")
    for v, (_, counts, usage, ppl) in results.items():
        print(f"   {v:3}{NAMES[v]:<28}{usage * 100:7.1f}%{ppl:12.1f}   {sparkline(counts)}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    from torchvision.utils import save_image

    rows = [x[:8]]
    with torch.no_grad():
        for model, *_ in results.values():
            rows.append(model(x[:8])[0])
    save_image((torch.cat(rows) + 1) / 2, out / "recon.png", nrow=8)
    print(f"\n   reconstructions saved to {out / 'recon.png'} (top: originals, then {', '.join(results)})")
    if not args.real:
        print("   (random decoder weights: images are noise; run with --real for faces)")


if __name__ == "__main__":
    main()
