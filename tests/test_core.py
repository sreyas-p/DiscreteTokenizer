"""Run: python -m pytest tests -q   (CPU, no downloads: uses a random-weight copy of the decoder)"""
import pytest
import torch

from tok.decoder import PretrainedDecoder, random_vqvae
from tok.metrics import CodeStats
from tok.model import Tokenizer
from tok.quantizers import Quantizer, VectorQuantizer


@pytest.fixture(scope="module")
def decoder():
    return PretrainedDecoder(random_vqvae())


def test_vq_nearest_code_loss_and_straight_through():
    torch.manual_seed(0)
    vq = VectorQuantizer(32, 8)
    vq.codebook.weight.data.normal_()
    z = torch.randn(2, 5, 8, requires_grad=True)
    z_q, idx, loss = vq(z)
    E = vq.codebook.weight
    assert torch.equal(idx, torch.cdist(z.reshape(-1, 8), E).argmin(-1).view(2, 5))
    assert torch.allclose(loss, (z - E[idx]).pow(2).mean() * 1.25, atol=1e-6)
    g = torch.randn_like(z_q)
    (z_q * g).sum().backward()
    assert torch.allclose(z.grad, g)  # gradient copied straight through to the encoder


@pytest.mark.parametrize("variant", list("ABCD"))
def test_shapes_bits_and_only_encoder_trains(decoder, variant):
    model = Tokenizer(decoder, variant)
    x = torch.rand(4, 3, 64, 64) * 2 - 1
    x_hat, latent, idx, aux = model(x)
    assert x_hat.shape == x.shape and latent.shape == (4, 3, 16, 16) and idx.shape == (4, 256)
    assert abs(model.bits_per_image() - 2560) < 10  # 256 tokens x 10 bits; FSQ: 1000 codes = 2551 bits
    (x_hat.mean() + aux).backward()
    assert all(p.grad is None for p in model.decoder.parameters())          # frozen
    assert model.encoder.net[0].weight.grad.abs().sum() > 0                 # trained through the decoder
    assert len(model.trainable_parameters()) < len(list(model.parameters()))


def test_decoder_stays_in_eval_mode(decoder):
    model = Tokenizer(decoder, "D").train()
    assert not model.decoder.vqvae.training and model.encoder.training


def test_quantizer_runs_fp32_under_autocast():
    q = Quantizer("A", 16, 64)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        z_q, _, loss = q(torch.randn(2, 8, 16).bfloat16())
    assert z_q.dtype == torch.float32 and loss.dtype == torch.float32


def test_code_stats():
    s = CodeStats(4, window=10)
    s.update(torch.tensor([0, 1, 2, 3]), step=1)
    assert s.usage() == 1.0 and abs(s.perplexity() - 4) < 1e-6
    s.reset_counts()
    s.update(torch.tensor([2, 2]), step=15)
    assert s.usage() == 0.25 and abs(s.perplexity() - 1) < 1e-6 and s.dead(15) == 3


@pytest.mark.parametrize("layout", ["official", "kaggle"])
def test_celeba_loader_reads_both_download_layouts(tmp_path, layout):
    from PIL import Image

    from tok.data import celeba

    names = [f"{i:06d}.jpg" for i in range(1, 7)]
    splits = [0, 0, 0, 1, 2, 2]
    if layout == "official":
        root, img_dir = tmp_path, tmp_path / "celeba" / "img_align_celeba"
        part = tmp_path / "celeba" / "list_eval_partition.txt"
        text = "".join(f"{n} {s}\n" for n, s in zip(names, splits))
    else:  # Kaggle: nested image folder, CSV with a header, no "celeba" folder
        root, img_dir = tmp_path, tmp_path / "img_align_celeba" / "img_align_celeba"
        part = tmp_path / "list_eval_partition.csv"
        text = "image_id,partition\n" + "".join(f"{n},{s}\n" for n, s in zip(names, splits))
    img_dir.mkdir(parents=True)
    part.write_text(text)
    for n in names:
        Image.new("RGB", (178, 218), (200, 100, 50)).save(img_dir / n)

    train, test = celeba(root, "train"), celeba(root, "test", eval_size=1)
    assert len(train) == 3 and len(test) == 1
    x = train[0]
    assert x.shape == (3, 64, 64) and -1 <= x.min() and x.max() <= 1
