# DiscreteTokenizer

**Software product:** a command-line research toolkit that trains and evaluates discrete image tokenizers.
**Tech stack:** Python 3.10+, PyTorch, torchvision, 🤗 Diffusers (pretrained VQ-VAE decoder), vector-quantize-pytorch,
LPIPS, TorchMetrics and pytest. Training runs on a CUDA GPU, and CPU works for tests and smoke runs. 

A small discrete image tokenizer for CelebA faces at 64×64. **Only the encoder is trained.** The decoder is a
frozen, pretrained **VQ-VAE** decoder: the VQ-f4 autoencoder released with Latent Diffusion and trained on
CelebA-HQ faces (see [Citations](#citations)).

```
x (3×64×64) ─► Encoder ─► 16×16×64 ─► Quantizer ─► 256 tokens ─► 1×1 conv ─► 16×16×3 ─► frozen VQ-VAE decoder ─► x̂
              [trained]              [trained]                    [trained]             [pretrained, not trained]
```

Our own quantizer produces the tokens. The pretrained VQ-VAE's codebook is skipped (`force_not_quantize=True`), and only
its decoder is used.

## Project plan

Models ‌like ‌VQGAN-transformers, ‌MaskGIT and Parti don't operate on pixels, they first tokenize each image into a short grid of discrete codes (similar to how words are tokens to a NLP language model). Tokenizers can be difficult to train, often suffering from codebook collapse where a few codes are used while the rest remain unused, wasting bits, resulting in blurry reconstructions. Multiple attempts have been made to fix this problem, but they are rarely benchmarked fairly. This project benchmarks them head-to-head on Celebrity images. A baseline VQ model (no fix), EMA with k-means initialization and dead code restarts, low-dimensional cosine codes, and FSQ. All ablations use the same image encoder, same frozen pretrained VQ-VAE decoder, same data and random seed, and same total number of bits (around 2560 per image). Using a frozen pretrained decoder decoder makes this project small and fair: only a ~1.6M encoder needs to be trained, any difference in results is due to the quantizer, and the decoder can be freely swapped out. For each model variant you will get a trained checkpoint, a row in `results/results.csv` (codebook usage, perplexity, number of dead codes, PSNR, SSIM, LPIPS), a grid of reconstructions, and a short answer to the question of which fix offers the best usage/quality balance. Potential future extensions include adding more model variants (LFQ, SimVQ), larger images, and training a small transformer to generate new faces.

## User roles

DiscreteTokenizer is a command-line toolkit, not a hosted web app. There are no accounts, nobody uploads content, and
nothing is served on the public internet, so moderation and login restrictions (e.g. `@ucsb.edu`) don't apply. It still
has four kinds of users:

| Role | Goal | How the toolkit supports it |
|---|---|---|
| **Researcher / student** (main user) | Find out which collapse fix works best at a fixed bitrate | `train.py --variant {A,B,C,D}` trains one tokenizer, `evaluate.py` adds a comparable row to `results/results.csv`, and `log.jsonl` tracks usage, perplexity and dead codes over training |
| **Generative-model builder** | Get discrete image tokens for a downstream model (e.g. a transformer that generates faces) | `Tokenizer.tokenize(x)` returns a `16×16` grid of token ids per image, and `Tokenizer(...)(x)` decodes back to pixels with the frozen decoder |
| **Reader / reviewer** (instructor, TA, classmate) | Check and reproduce the results | One command per run, seed 0 everywhere, and a fixed eval split (first 10k CelebA test images). `pytest` checks the core math on CPU with no downloads, and `--smoke` checks the full pipeline in about a minute |
| **Contributor** | Add a new quantizer and compare it fairly | Implement the one-method interface in `tok/quantizers.py` (`z → z_q, ids, aux_loss`). The tests check shapes, the bit budget and that the decoder stays frozen |

## Current experience

What the pipeline does:

What a user sees today: this is real output from the built-in smoke test (`--smoke` uses random images and a
random-weight copy of the decoder, so the numbers only show that the pipeline runs and are not results):

```text
$ python train.py --variant B --smoke --steps 20 --log-every 10 --lpips-weight 0 --batch-size 16
variant B: 2560.0 bits/image, 1.59M trainable params, device cpu
step     10 | loss 0.4813 | usage 1.000 | ppl   288.0 | dead    0
step     20 | loss 0.4418 | usage 0.177 | ppl    97.4 | dead    0
saved runs/B/last.pt

$ python evaluate.py --ckpt runs/B/last.pt --eval-size 32 --no-lpips
{'variant': 'B', 'steps': 20, 'bits_per_image': '2560.0', 'usage': '0.0967', 'perplexity': '54.8', 'psnr': '12.188', 'ssim': '0.1770', 'lpips': ''}

$ cat results/results.csv
variant,steps,bits_per_image,usage,perplexity,psnr,ssim,lpips
B,20,2560.0,0.0967,54.8,12.188,0.1770,
```

Each evaluation also saves `results/recon_<variant>.png`, an 8-image grid with the originals on top and the
reconstructions below. Status: the code, tests and smoke runs are complete. The full CelebA runs (30k steps per variant)
need a GPU and haven't been done yet.

## Tech Stack

| Technology | What it's used for | Why this choice |
|---|---|---|
| **Python 3.10+** | Everything | The standard language for ML research. Every library below is Python-first |
| **PyTorch** (+ torchvision) | Encoder, training loop, CelebA loading | Dynamic graphs make straight-through estimators (`z + (z_q - z).detach()`) easy to write and debug. torchvision ships a CelebA loader with the official train/test split |
| **🤗 Diffusers** | Loads the pretrained VQ-VAE (`VQModel`) | One line downloads a published, peer-reviewed decoder, so we don't train (and justify) our own |
| **vector-quantize-pytorch** (pinned 1.31.6) | Quantizer variants B–D | Widely used, well-tested implementations of EMA/k-means VQ, cosine codes and FSQ. Pinned because its API changes often. Variant A is written by hand so the core math is visible |
| **LPIPS** | Perceptual training loss and metric | The standard perceptual metric for tokenizers. It matches human judgement better than pixel error |
| **TorchMetrics** | SSIM | A tested reference implementation, so we avoid metric bugs |
| **pytest** | Unit tests | Fast CPU tests for the math (nearest-code lookup, loss, gradient flow) and the fairness rules |
| **Git + GitHub** | Version control, tagged releases | Reproducibility: each result can be traced to a tagged commit |

We considered and rejected two alternatives. **JAX/Flax** has fewer pretrained tokenizers and VQ libraries. **Training
our own decoder** (the original plan) would have multiplied compute, and decoder quality would have confounded the
quantizer comparison.

## License

Code: MIT, see [LICENSE.md](LICENSE.md). The pretrained decoder weights and the CelebA dataset keep their own licenses
and terms. CelebA is available for non-commercial research purposes only.

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

## Files

```
tok/data.py        CelebA loader (64×64, [-1, 1])
tok/model.py       encoder (×4 downsampling) + quantizer + projection in front of the frozen decoder
tok/quantizers.py  variants A–D behind one interface
tok/decoder.py     the borrowed pretrained VQ-VAE decoder (frozen)
tok/metrics.py     usage, perplexity, dead codes, PSNR
train.py, evaluate.py, tests/test_core.py
```

## Citations

**Decoder (pretrained, used frozen).** The decoder of the VQ-regularized autoencoder (VQ-f4, 8192 codes) from
[`CompVis/ldm-celebahq-256`](https://huggingface.co/CompVis/ldm-celebahq-256) (subfolder `vqvae`), trained on
CelebA-HQ and loaded with [🤗 Diffusers](https://github.com/huggingface/diffusers). Its architecture follows VQGAN
(Esser et al., 2021).

```bibtex
@inproceedings{rombach2022ldm,
  title     = {High-Resolution Image Synthesis with Latent Diffusion Models},
  author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {10684--10695},
  year      = {2022}
}

@inproceedings{esser2021taming,
  title     = {Taming Transformers for High-Resolution Image Synthesis},
  author    = {Esser, Patrick and Rombach, Robin and Ommer, Bj{\"o}rn},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {12873--12883},
  year      = {2021}
}
```

**Dataset.** CelebA, with CelebA-HQ (Karras et al., *Progressive Growing of GANs*, ICLR 2018) for the decoder's pretraining.

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
