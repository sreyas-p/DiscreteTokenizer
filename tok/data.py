"""CelebA at 64x64 in [-1, 1].

Liu, Luo, Wang, Tang. "Deep Learning Face Attributes in the Wild." ICCV 2015.
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Works with either download, placed anywhere under <root>/celeba (or directly under <root>):
  - official / torchvision:  img_align_celeba/*.jpg + list_eval_partition.txt
  - Kaggle (jessicali9530/celeba-dataset):  img_align_celeba/img_align_celeba/*.jpg + list_eval_partition.csv
Splits follow the official partition file (0 = train, 1 = valid, 2 = test). The eval set is the first
`eval_size` images of the test split and is never trained on.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

SPLITS = {"train": 0, "valid": 1, "test": 2}


def _find(base, pattern):
    hits = sorted(base.rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"no {pattern} under {base}; see the README for where to put CelebA")
    return hits[0]


def read_partition(path):
    """list_eval_partition.txt ("000001.jpg 0") or .csv ("image_id,partition" header) -> [(name, split)]."""
    rows = []
    for line in Path(path).read_text().splitlines():
        parts = line.replace(",", " ").split()
        if len(parts) == 2 and parts[1].isdigit():
            rows.append((parts[0], int(parts[1])))
    return rows


class CelebA(Dataset):
    def __init__(self, root="data", split="train", image_size=64, limit=None):
        base = Path(root) / "celeba" if (Path(root) / "celeba").is_dir() else Path(root)
        partition = next((p for p in sorted(base.rglob("list_eval_partition.*")) if p.suffix in (".txt", ".csv")),
                         None)
        if partition is None:
            raise FileNotFoundError(f"no list_eval_partition.txt/.csv under {base}")
        self.image_dir = _find(base, "000001.jpg").parent
        self.files = [name for name, s in read_partition(partition) if s == SPLITS[split]][:limit]
        self.tf = T.Compose([T.CenterCrop(178), T.Resize(image_size), T.ToTensor(),
                             T.Normalize([0.5] * 3, [0.5] * 3)])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.image_dir / self.files[i]).convert("RGB"))


def celeba(root="data", split="train", image_size=64, eval_size=10000):
    return CelebA(root, split, image_size, limit=eval_size if split == "test" else None)


class RandomImages(Dataset):
    """Smooth random images, only for tests and --smoke runs (no download needed)."""

    def __init__(self, n, image_size=64, seed=0):
        self.n, self.size, self.seed = n, image_size, seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        x = torch.rand(1, 3, 4, 4, generator=g)
        return torch.nn.functional.interpolate(x, size=self.size, mode="bilinear")[0] * 2 - 1
