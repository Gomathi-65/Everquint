"""Data pipeline: CIFAR-10 download, normalisation, augmentation, train/val split, DataLoaders."""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from utils import MEAN, STD, seed_worker


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # random shifts
        transforms.RandomHorizontalFlip(),      # mirror images
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),        # per-channel mean/std of the training set
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return train_tf, eval_tf


def get_loaders(data_dir="data", batch_size=128, val_size=5000, num_workers=2, seed=42, fake=False):
    """Returns train_loader, val_loader, test_loader.

    The 50k training images are split into 45k train / 5k validation with a seeded permutation.
    The validation set uses the non-augmented transform. fake=True uses random images (smoke tests only).
    """
    train_tf, eval_tf = get_transforms()
    if fake:
        mk = lambda n, tf, off: datasets.FakeData(size=n, image_size=(3, 32, 32), num_classes=10,
                                                  transform=tf, random_offset=off)
        train_full, eval_full, test_set = mk(1000, train_tf, 0), mk(1000, eval_tf, 0), mk(200, eval_tf, 10_000)
        val_size = 100
    else:
        train_full = datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        eval_full = datasets.CIFAR10(data_dir, train=True, download=False, transform=eval_tf)
        test_set = datasets.CIFAR10(data_dir, train=False, download=True, transform=eval_tf)

    perm = torch.randperm(len(train_full), generator=torch.Generator().manual_seed(seed)).tolist()
    train_set, val_set = Subset(train_full, perm[val_size:]), Subset(eval_full, perm[:val_size])

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
              persistent_workers=num_workers > 0, worker_init_fn=seed_worker)
    train_loader = DataLoader(train_set, shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(seed), **kw)
    val_loader = DataLoader(val_set, shuffle=False, **kw)
    test_loader = DataLoader(test_set, shuffle=False, **kw)
    return train_loader, val_loader, test_loader
