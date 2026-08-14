# src/data/transforms.py
# Data augmentations and normalization.

import torch
from torchvision import transforms

# CIFAR-10 Augmentations
# As per requirements: "Use only RandAugment for data augmentation"
# Normalization values for CIFAR-10: mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010)

def get_cifar10_train_transforms(randaugment_num_ops=2, randaugment_magnitude=9):
    """
    Returns training transforms for CIFAR-10: RandAugment, ToTensor, Normalize.
    """
    return transforms.Compose([
        transforms.RandAugment(num_ops=randaugment_num_ops, magnitude=randaugment_magnitude),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

def get_cifar10_test_transforms():
    """
    Returns testing transforms for CIFAR-10: ToTensor, Normalize.
    No augmentation for testing.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])


# ModelNet40 Augmentations and Normalization
# Requirements:
# - Only apply per-point scaling as augmentation
# - Normalize to unit cube and zero-center

def normalize_point_cloud(points):
    """
    Normalize point cloud to be within a unit cube and zero-centered.
    Args:
        points (torch.Tensor): Point cloud of shape (num_points, 3).
    Returns:
        torch.Tensor: Normalized point cloud.
    """
    centroid = torch.mean(points, axis=0)
    points = points - centroid
    max_dist = torch.max(torch.sqrt(torch.sum(points ** 2, axis=1)))
    if max_dist > 0: # Avoid division by zero for empty or single-point clouds
        points = points / max_dist
    return points

def scale_point_cloud(points, scale_min=0.8, scale_max=1.2):
    """
    Apply per-point random scaling.
    This usually means scaling the entire object, not individual points independently.
    Let's assume scaling the entire object.
    Args:
        points (torch.Tensor): Point cloud of shape (num_points, 3).
        scale_min (float): Minimum scaling factor.
        scale_max (float): Maximum scaling factor.
    Returns:
        torch.Tensor: Scaled point cloud.
    """
    scale = torch.rand(1) * (scale_max - scale_min) + scale_min
    return points * scale

def translate_point_cloud(points, translate_range=0.02):
    """
    Apply random translation to point cloud.
    Args:
        points (torch.Tensor): Point cloud of shape (num_points, 3).
        translate_range (float): Translation range [-translate_range, +translate_range].
    Returns:
        torch.Tensor: Translated point cloud.
    """
    translation = (torch.rand(3) - 0.5) * 2 * translate_range  # Range [-translate_range, +translate_range]
    return points + translation

def rotate_point_cloud(points):
    """
    Apply random rotation to point cloud around a random axis.
    Args:
        points (torch.Tensor): Point cloud of shape (num_points, 3).
    Returns:
        torch.Tensor: Rotated point cloud.
    """
    # Random rotation angle
    angle = torch.rand(1) * 2 * 3.14159  # Random angle [0, 2π]
    
    # Random rotation axis (normalized)
    axis = torch.randn(3)
    axis = axis / torch.norm(axis)
    
    # Rodrigues' rotation formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Cross product matrix for the axis
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]], 
                      [-axis[1], axis[0], 0]])
    
    # Rotation matrix using Rodrigues' formula
    R = torch.eye(3) + sin_angle * K + (1 - cos_angle) * torch.matmul(K, K)
    
    # Apply rotation
    return torch.matmul(points, R.T)

class ModelNet40Augmentation:
    def __init__(self, augment=True, scale_min=0.8, scale_max=1.2, 
                 use_translation=False, translate_range=0.02,
                 use_rotation=False):
        self.augment = augment
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.use_translation = use_translation
        self.translate_range = translate_range
        self.use_rotation = use_rotation

    def __call__(self, points):
        """
        Args:
            points (np.array or torch.Tensor): Point cloud, shape (num_points, 3).
        Returns:
            torch.Tensor: Processed point cloud.
        """
        if not isinstance(points, torch.Tensor):
            points = torch.from_numpy(points).float()

        if self.augment:
            # Apply scaling (always applied when augment=True)
            points = scale_point_cloud(points, self.scale_min, self.scale_max)
            
            # Apply translation if enabled
            if self.use_translation:
                points = translate_point_cloud(points, self.translate_range)
            
            # Apply rotation if enabled
            if self.use_rotation:
                points = rotate_point_cloud(points)
        
        points = normalize_point_cloud(points) # Always normalize
        return points
