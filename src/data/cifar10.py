# src/data/cifar10.py
# CIFAR-10 dataset wrapper for the Perceiver model.

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T
from ..utils.positional_encoding import FourierPositionalEncoding
from .transforms import get_cifar10_train_transforms, get_cifar10_test_transforms

class CIFAR10PerceiverDataModule:
    def __init__(self, 
                 data_dir='./data', 
                 batch_size=64,
                 num_workers=4,
                 input_dim=196,  # (patch_size=2, embedding_dim=32*3 = 96 => 2*2*96 = 384)
                 image_size=32,
                 patch_size=2,
                 fourier_dim=128,
                 circular_pos_encoding=True,
                 max_frequencies=10,
                 num_frequency_bands=6,
                 randaugment_num_ops=2,
                 randaugment_magnitude=9,
                 flatten_patches=True):
        """
        DataModule for CIFAR-10 dataset with positional encoding and patch embedding.
        
        Args:
            data_dir: Path to data directory
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            input_dim: Dimension of the input to the Perceiver model
            image_size: Size of the input images
            patch_size: Size of image patches
            fourier_dim: Dimension of Fourier positional encoding
            circular_pos_encoding: Whether to use circular positional encoding
            max_frequencies: Maximum frequency for Fourier encoding
            num_frequency_bands: Number of frequency bands for Fourier encoding
            randaugment_num_ops: Number of operations for RandAugment
            randaugment_magnitude: Magnitude of operations for RandAugment
            flatten_patches: Whether to flatten patches or keep spatial dimensions
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_dim = input_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.flatten_patches = flatten_patches
        self.randaugment_num_ops = randaugment_num_ops
        self.randaugment_magnitude = randaugment_magnitude
        self.circular_pos_encoding = circular_pos_encoding
        self.fourier_dim = fourier_dim
        self.max_frequencies = max_frequencies
        self.num_frequency_bands = num_frequency_bands
    
        # Calculate number of patches
        self.num_patches = (self.image_size // self.patch_size) ** 2
        
        # Setup positional encoding
        self._setup_pos_encoding()
        
        # Create transforms based on parameters
        self.train_transform = get_cifar10_train_transforms(
            randaugment_num_ops=self.randaugment_num_ops,
            randaugment_magnitude=self.randaugment_magnitude
        )
        
        self.test_transform = get_cifar10_test_transforms()

    def _setup_pos_encoding(self):
        """Setup positional encoding for patches"""
        # Calculate grid size based on image size and patch size
        grid_size = self.image_size // self.patch_size
        
        # Create positional encoding
        self.pos_encoding = FourierPositionalEncoding(
            dim=self.fourier_dim,
            max_spatial_size=grid_size,
            max_frequencies=self.max_frequencies,
            num_frequency_bands=self.num_frequency_bands,
            circular=self.circular_pos_encoding
        )
        
        # Create a grid of coordinates for the patches
        grid_h = torch.arange(grid_size)
        grid_w = torch.arange(grid_size)
        coords = torch.stack(torch.meshgrid([grid_h, grid_w], indexing='ij'), dim=-1)
        
        # Normalize coordinates based on grid size
        norm_coords = coords.float() / (grid_size - 1)
        if not self.circular_pos_encoding:
            # Rescale from [0, 1] to [-1, 1] for non-circular encoding
            norm_coords = norm_coords * 2 - 1
            
        # Reshape to match expected format: [num_patches, num_dimensions]
        # Each patch has 2 coordinates (h, w)
        self.patch_coord_indices = norm_coords.reshape(-1, 2)
        
        # Pre-compute positional encodings for all patches
        with torch.no_grad():
            self.patch_pos_encodings = self.pos_encoding(self.patch_coord_indices)
            # [num_patches, fourier_dim]
    
    def _to_patches(self, x):
        """
        Convert a batch of images to patches.
        Args:
            x: [batch_size, channels, height, width]
        Returns:
            Tensor of shape [batch_size, num_patches, patch_dim]
        """
        batch_size = x.size(0)
        channels = x.size(1)
        
        # Reshape into patches [B, C, H//P, P, W//P, P]
        patches = x.view(
            batch_size, channels,
            self.image_size // self.patch_size, self.patch_size,
            self.image_size // self.patch_size, self.patch_size
        )
        
        # Permute to [B, H//P, W//P, P, P, C]
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        
        if self.flatten_patches:
            # Flatten each patch and combine spatial dimensions
            # [B, H//P, W//P, P*P*C]
            patches = patches.reshape(batch_size, -1, self.patch_size * self.patch_size * channels)
        else:
            # Keep spatial structure of each patch
            # [B, H//P, W//P, P, P, C]
            patches = patches.reshape(batch_size, -1, self.patch_size, self.patch_size, channels)
        
        return patches
    
    def _add_pos_encoding(self, patches):
        """
        Add positional encoding to patches.
        Args:
            patches: [batch_size, num_patches, patch_dim]
        Returns:
            Tensor of shape [batch_size, num_patches, patch_dim + fourier_dim]
        """
        batch_size = patches.size(0)
        
        # Expand positional encoding to batch size
        # [1, num_patches, fourier_dim] -> [batch_size, num_patches, fourier_dim]
        pos_encodings = self.patch_pos_encodings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Flatten patch dimensions if needed
        if not self.flatten_patches:
            patches = patches.reshape(batch_size, -1, self.patch_size * self.patch_size * 3)
        
        # Concatenate along feature dimension
        # [batch_size, num_patches, patch_dim + fourier_dim]
        patches_with_pos = torch.cat([patches, pos_encodings.to(patches.device)], dim=2)
        
        return patches_with_pos
    
    def setup(self):
        """Set up the CIFAR-10 dataset."""
        # Download and setup CIFAR-10 datasets
        self.train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transform
        )
        
        self.val_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transform
        )
    
    def train_dataloader(self):
        """Create training DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def val_dataloader(self):
        """Create validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
    
    def preprocess_batch(self, batch):
        """
        Preprocess a batch of data.
        
        Args:
            batch: Tuple of (images, labels)
        
        Returns:
            Dictionary containing processed inputs and labels
        """
        images, labels = batch
        
        # Convert images to patches
        patches = self._to_patches(images)
        
        # Add positional encoding
        patches_with_pos = self._add_pos_encoding(patches)
        
        # Return dictionary with processed batch
        return {
            'inputs': patches_with_pos,
            'labels': labels,
            'original_images': images  # Keep original for visualization
        }
