# src/data/cifar10.py
# CIFAR-10 per il Perceiver: token grezzi, nessuna positional encoding.
# La PE vive nel modello (src/perceiver/input_pe.py).

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from .transforms import get_cifar10_test_transforms, get_cifar10_train_transforms


class CIFAR10PerceiverDataModule:
    def __init__(
        self,
        data_dir="./data",
        batch_size=64,
        num_workers=4,
        image_size=32,
        patch_size=1,
        val_split=5000,
        split_seed=42,
        randaugment_num_ops=2,
        randaugment_magnitude=9,
    ):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.patch_size = patch_size
        self.val_split = val_split
        self.split_seed = split_seed

        if image_size % patch_size != 0:
            raise ValueError(f"image_size {image_size} non divisibile per patch_size {patch_size}")

        self.num_patches = (image_size // patch_size) ** 2
        self.token_dim = patch_size * patch_size * 3

        self.train_transform = get_cifar10_train_transforms(
            randaugment_num_ops=randaugment_num_ops,
            randaugment_magnitude=randaugment_magnitude,
        )
        self.test_transform = get_cifar10_test_transforms()

        self.train_indices = []
        self.val_indices = []

    def _to_patches(self, x):
        """(B, 3, H, W) -> (B, num_patches, token_dim). Con patch_size=1 sono pixel grezzi."""
        batch_size, channels = x.size(0), x.size(1)
        grid = self.image_size // self.patch_size
        patches = x.view(batch_size, channels, grid, self.patch_size, grid, self.patch_size)
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        return patches.reshape(batch_size, self.num_patches, self.token_dim)

    def setup(self):
        train_full = CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.train_transform
        )
        # Stessi dati, ma senza augmentation: serve per la validation split.
        val_full = CIFAR10(
            root=self.data_dir, train=True, download=True, transform=self.test_transform
        )

        generator = torch.Generator().manual_seed(self.split_seed)
        permutation = torch.randperm(len(train_full), generator=generator).tolist()

        self.val_indices = permutation[: self.val_split]
        self.train_indices = permutation[self.val_split :]

        self.train_dataset = Subset(train_full, self.train_indices)
        self.val_dataset = Subset(val_full, self.val_indices)
        self.test_dataset = CIFAR10(
            root=self.data_dir, train=False, download=True, transform=self.test_transform
        )

    def _loader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)

    def preprocess_batch(self, batch):
        images, labels = batch
        return {
            "inputs": self._to_patches(images),
            "labels": labels,
            "original_images": images,
        }
