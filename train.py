"""Train the encoder + quantizer in front of the frozen pretrained decoder.

    python train.py --variant B
    python train.py --variant A --steps 2000 --smoke     # random data + random decoder, no downloads

Loss = L1(x_hat, x) + lpips_weight * LPIPS(x_hat, x)          reconstruction through the frozen decoder
     + latent_weight * MSE(latent, teacher latent)            match the pretrained encoder's latent
     + quantizer loss (codebook / commitment, 0 for FSQ)
Only the encoder, quantizer and the 1x1 projection are trained.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tok.data import RandomImages, celeba
from tok.decoder import PretrainedDecoder, random_vae
from tok.metrics import CodeStats
from tok.model import Tokenizer


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="B", choices=list("ABCD"))
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=64)  # 64 x 64 positions >= 1024 codes for k-means init
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--latent-weight", type=float, default=1.0)
    ap.add_argument("--lpips-weight", type=float, default=1.0)
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=None, help="default: runs/<variant>")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="random images + random-weight decoder (pipeline check)")
    return ap.parse_args()


def main():
    args = parse()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out or f"runs/{args.variant}")
    out.mkdir(parents=True, exist_ok=True)

    data = RandomImages(1024) if args.smoke else celeba(args.data_root, "train")
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=True,
                        num_workers=0 if args.smoke else args.num_workers,
                        generator=torch.Generator().manual_seed(args.seed))

    decoder = PretrainedDecoder(random_vae() if args.smoke else None)
    model = Tokenizer(decoder, args.variant).to(device)
    lpips_fn = None
    if args.lpips_weight > 0:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(device).eval().requires_grad_(False)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=args.lr, weight_decay=0.0)
    stats = CodeStats(model.quantizer.num_codes)

    print(f"variant {args.variant}: {model.bits_per_image():.1f} bits/image, "
          f"{sum(p.numel() for p in model.trainable_parameters()) / 1e6:.2f}M trainable params, device {device}")
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    log = open(out / "log.jsonl", "a")
    step, sums, t0 = 0, {}, time.time()
    model.train()
    while step < args.steps:
        for x in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(**amp):
                x_hat, latent, idx, q_loss = model(x)
                target = decoder.teacher_latents(x)
            terms = {"l1": (x_hat.float() - x).abs().mean(),
                     "latent": F.mse_loss(latent.float(), target.float()),
                     "quantizer": q_loss}
            if lpips_fn is not None:
                terms["lpips"] = lpips_fn(x_hat.float(), x).mean()
            loss = terms["l1"] + args.latent_weight * terms["latent"] + terms["quantizer"] + \
                args.lpips_weight * terms.get("lpips", 0.0)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), 1.0)
            opt.step()
            step += 1

            stats.update(idx, step)
            for k, v in {"loss": loss, **terms}.items():
                sums[k] = sums.get(k, 0.0) + float(v.detach())
            if step % args.log_every == 0 or step == args.steps:
                n = args.log_every if step % args.log_every == 0 else step % args.log_every
                row = {"step": step, **{k: v / n for k, v in sums.items()}, "usage": stats.usage(),
                       "perplexity": stats.perplexity(), "dead_codes": stats.dead(step),
                       "sec_per_step": (time.time() - t0) / n}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"step {step:6d} | loss {row['loss']:.4f} | usage {row['usage']:.3f} | "
                      f"ppl {row['perplexity']:7.1f} | dead {row['dead_codes']:4d}", flush=True)
                stats.reset_counts()
                sums, t0 = {}, time.time()
            if step % args.save_every == 0 or step == args.steps:
                # the frozen decoder is not saved; it is reloaded from the hub
                state = {k: v for k, v in model.state_dict().items() if not k.startswith("decoder.")}
                torch.save({"model": state, "variant": args.variant, "step": step, "smoke": args.smoke}, out / "last.pt")
            if step >= args.steps:
                break
    print(f"saved {out / 'last.pt'}")


if __name__ == "__main__":
    main()
