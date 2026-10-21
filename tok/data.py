"""CelebA at 64x64 in [-1, 1].

Liu, Luo, Wang, Tang. "Deep Learning Face Attributes in the Wild." ICCV 2015.
https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Expected layout (torchvision): <root>/celeba/{img_align_celeba/, list_eval_partition.txt, ...}
The eval set is the first `eval_size` images of the official test split and is never trained on.
"""
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets
from torchvision import transforms as T


class ImagesOnly(Dataset):
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0]


def celeba(root="data", split="train", image_size=64, eval_size=10000, download=False):
    tf = T.Compose([T.CenterCrop(178), T.Resize(image_size), T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    ds = datasets.CelebA(root, split=split, target_type=[], transform=tf, download=download)
    if split == "test":
        ds = Subset(ds, range(min(eval_size, len(ds))))
    return ImagesOnly(ds)


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
