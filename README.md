# DiscreteTokenizer

A small discrete image tokenizer for CelebA faces at 64×64. **Only the encoder is trained.** The decoder is a
frozen, pretrained decoder borrowed from Stable Diffusion's VAE (see [Citations](#citations)).

```
x (3×64×64) ─► Encoder ─► 8×8×64 ─► Quantizer ─► 64 tokens ─► 1×1 conv ─► 8×8×4 ─► frozen SD-VAE decoder ─► x̂
              [trained]            [trained]                  [trained]           [pretrained, not trained]
```

**Question:** at a fixed budget of about 640 bits per image (64 tokens × ~10 bits), how do four quantizers
trade off codebook usage against reconstruction quality?

| Variant | Quantizer | Codes |
|---|---|---|
| A | plain VQ with a loss-trained codebook (baseline, expected to collapse) | 1024 |
| B | VQ + EMA + k-means init + dead-code restart | 1024 |
| C | low-dimensional (8) L2-normalized codes | 1024 |
| D | FSQ, levels [8, 5, 5, 5] | 1000 |

## Setup

```bash
pip install -r requirements.txt
python -m pytest tests -q          # CPU, no downloads
```

CelebA goes in `data/celeba/` (torchvision layout: `img_align_celeba/`, `list_eval_partition.txt`, and the other
annotation files). Download it from the [CelebA page](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
Images are center-cropped to 178×178 and resized to 64×64. Evaluation uses the first 10,000 images of the official
test split, which are never trained on. The pretrained decoder downloads from Hugging Face on first use.

## Run

```bash
python train.py --variant B                 # 30k steps, batch 64, AdamW lr 2e-4, seed 0
python evaluate.py --ckpt runs/B/last.pt    # adds a row to results/results.csv + results/recon_B.png

for v in A B C D; do python train.py --variant $v && python evaluate.py --ckpt runs/$v/last.pt; done

python train.py --variant A --smoke --steps 50 --lpips-weight 0   # pipeline check: random data + random decoder
```

Every variant uses the same data, steps, seed, encoder, decoder and bit budget. Only the quantizer changes.

**Training loss:**

- **reconstruction through the frozen decoder:** `L1(x̂, x) + 1.0 · LPIPS(x̂, x)`
- **latent matching:** `1.0 · MSE(latent, teacher latent)`. The teacher latent is the pretrained SD-VAE encoder's latent mean. It tells our encoder roughly where the decoder expects its inputs, which speeds training up a lot. Set `--latent-weight 0` to train from reconstruction alone.
- **quantizer loss:** codebook and commitment terms with β = 0.25. FSQ has none.

`runs/<variant>/log.jsonl` records every loss term every 200 steps, along with codebook usage, perplexity and dead
codes (codes unused for 1000 steps).

`results/results.csv` columns: `variant, steps, bits_per_image, usage, perplexity, psnr, ssim, lpips`.

## Files

```
tok/data.py        CelebA loader (64×64, [-1, 1])
tok/model.py       encoder + quantizer + projection in front of the frozen decoder
tok/quantizers.py  variants A–D behind one interface
tok/decoder.py     the borrowed pretrained decoder (frozen)
tok/metrics.py     usage, perplexity, dead codes, PSNR
train.py, evaluate.py, tests/test_core.py
```

## Citations

**Decoder (pretrained, used frozen).** The decoder of the Stable Diffusion VAE, fine-tuned with an MSE-weighted
loss: [`stabilityai/sd-vae-ft-mse`](https://huggingface.co/stabilityai/sd-vae-ft-mse) (MIT license), loaded with
[🤗 Diffusers](https://github.com/huggingface/diffusers).

```bibtex
@inproceedings{rombach2022ldm,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10684--10695},
  year      = {2022}
}
```

**Dataset.**

```bibtex
@inproceedings{liu2015celeba,
  title     = {Deep Learning Face Attributes in the Wild},
  author    = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of the International Conference on Computer Vision (ICCV)},
  year      = {2015}
}
```

**Quantizers and metrics.**

- VQ-VAE: van den Oord, Vinyals, Kavukcuoglu. *Neural Discrete Representation Learning.* NeurIPS 2017, arXiv:1711.00937
- Low-dimensional, L2-normalized codes: Yu et al. *Vector-quantized Image Modeling with Improved VQGAN.* ICLR 2022, arXiv:2110.04627
- FSQ: Mentzer, Minnen, Agustsson, Tschannen. *Finite Scalar Quantization: VQ-VAE Made Simple.* ICLR 2024, arXiv:2309.15505
- Variants B–D use [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch) v1.31.6
- LPIPS: Zhang, Isola, Efros, Shechtman, Wang. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018
