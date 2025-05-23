# src/data/modelnet40.py
# ModelNet40 dataset wrapper for the Perceiver model.

import os
import torch
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch.utils.data import Dataset
from ..utils.positional_encoding import FourierPositionalEncoding
from .transforms import ModelNet40Augmentation, normalize_point_cloud

class ModelNet40PerceiverDataModule:
    def __init__(self, 
                 data_dir='./data', 
                 batch_size=32,
                 num_workers=4,
                 num_points=1024,
                 fourier_dim=128,
                 max_frequencies=10,
                 num_frequency_bands=6,
                 augment_train=True):
        """
        DataModule for ModelNet40 dataset with point cloud processing for the Perceiver model.
        
        Args:
            data_dir: Path to data directory
            batch_size: Batch size
            num_workers: Number of workers for DataLoader
            num_points: Number of points per point cloud
            fourier_dim: Dimension of Fourier positional encoding
            max_frequencies: Maximum frequency for Fourier encoding
            num_frequency_bands: Number of frequency bands for Fourier encoding
            augment_train: Whether to apply data augmentation to the training set
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_points = num_points
        self.fourier_dim = fourier_dim
        self.max_frequencies = max_frequencies
        self.num_frequency_bands = num_frequency_bands
        self.augment_train = augment_train
        
        # Set up positional encoding
        self._setup_pos_encoding()
        
        # Create transforms
        self.train_transform = T.Compose([
            T.SamplePoints(num_points),
            T.NormalizeScale(),
        ])
        
        self.test_transform = T.Compose([
            T.SamplePoints(num_points),
            T.NormalizeScale(),
        ])
        
        # Point cloud augmentation (separate from torch_geometric transforms)
        self.train_augmentation = ModelNet40Augmentation(augment=augment_train)
        self.test_augmentation = ModelNet40Augmentation(augment=False)
        
        # ModelNet40 class names for reference
        self.class_names = [
            'airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair',
            'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box',
            'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person',
            'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table',
            'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
        ]
        
        # Number of classes
        self.num_classes = len(self.class_names)

    def _setup_pos_encoding(self):
        """Setup positional encoding for 3D points"""
        # For point clouds, we use 3D coordinates
        # Create positional encoding
        self.pos_encoding = FourierPositionalEncoding(
            dim=self.fourier_dim,
            max_spatial_size=1.0,  # Point clouds are normalized to unit cube
            max_frequencies=self.max_frequencies,
            num_frequency_bands=self.num_frequency_bands,
            num_pos_feats=3  # 3D coordinates (x,y,z)
        )
    
    def _process_point_cloud(self, points, augmentation):
        """
        Process a point cloud for input to the Perceiver model.
        
        Args:
            points: Point cloud tensor of shape [N, 3]
            augmentation: ModelNet40Augmentation instance
            
        Returns:
            Tensor of shape [N, 3 + fourier_dim] with positional encoding
        """
        # Apply augmentation
        points = augmentation(points)
        
        # Compute positional encodings
        pos_enc = self.pos_encoding(points)
        
        # Concatenate point coordinates and positional encoding
        return torch.cat([points, pos_enc], dim=-1)
    
    def setup(self):
        """Set up the ModelNet40 dataset."""
        # Download and setup ModelNet40 datasets
        self.train_dataset = ModelNet(
            root=os.path.join(self.data_dir, 'modelnet40'),
            name='40',
            train=True,
            transform=self.train_transform
        )
        
        self.val_dataset = ModelNet(
            root=os.path.join(self.data_dir, 'modelnet40'),
            name='40',
            train=False,
            transform=self.test_transform
        )
        
        # Wrap with custom dataset for Perceiver-specific processing
        self.train_dataset = ModelNet40PerceiverWrapper(
            self.train_dataset, 
            self.train_augmentation,
            self.pos_encoding
        )
        
        self.val_dataset = ModelNet40PerceiverWrapper(
            self.val_dataset,
            self.test_augmentation,
            self.pos_encoding
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
        This method handles PyTorch Geometric's batching which concatenates 
        point clouds into a single tensor. We need to split it back into 
        proper batch format.
        
        Args:
            batch: PyG Data object
        
        Returns:
            Dictionary containing processed inputs and labels
        """
        # PyTorch Geometric batches point clouds by concatenating them
        # We need to split them back into [batch_size, num_points, features]
        
        # Get batch information
        batch_size = batch.y.shape[0]
        total_points = batch.pos_with_encoding.shape[0]
        feature_dim = batch.pos_with_encoding.shape[1]
        num_points_per_sample = total_points // batch_size
        
        # Reshape from [total_points, features] to [batch_size, num_points, features]
        inputs = batch.pos_with_encoding.view(batch_size, num_points_per_sample, feature_dim)
        labels = batch.y
        
        # Also reshape original points for visualization
        original_points = batch.pos.view(batch_size, num_points_per_sample, 3)
        
        # Return dictionary with processed batch
        return {
            'inputs': inputs,
            'labels': labels,
            'original_points': original_points
        }


class ModelNet40PerceiverWrapper(Dataset):
    """
    Wrapper dataset for ModelNet40 that applies Perceiver-specific processing.
    """
    def __init__(self, dataset, augmentation, pos_encoding):
        """
        Args:
            dataset: PyG ModelNet dataset
            augmentation: ModelNet40Augmentation instance
            pos_encoding: FourierPositionalEncoding instance
        """
        self.dataset = dataset
        self.augmentation = augmentation
        self.pos_encoding = pos_encoding
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        # Apply augmentation to points
        points = self.augmentation(data.pos)
        
        # Compute positional encodings without gradients to avoid collation issues
        with torch.no_grad():
            pos_enc = self.pos_encoding(points).detach()
        
        # Concatenate point coordinates and positional encoding
        data.pos_with_encoding = torch.cat([points, pos_enc], dim=-1)
        
        return data
