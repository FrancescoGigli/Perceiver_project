# src/config/base_cfg.py
# Centralized configurations for datasets, model, and training.

import argparse
import torch

def get_base_config():
    parser = argparse.ArgumentParser(description="Base Configuration for Perceiver Project")

    # Dataset configurations
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'modelnet40'],
                        help='Dataset to use (cifar10 or modelnet40)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing datasets')
    parser.add_argument('--cifar10_fourier_bands', type=int, default=64, help='Number of Fourier bands for CIFAR-10')
    parser.add_argument('--cifar10_max_freq', type=float, default=32.0, help='Max frequency for CIFAR-10 Fourier')
    parser.add_argument('--permute_pixels', action='store_true', help='Enable permutation of CIFAR-10 pixels after PE')
    parser.add_argument('--permute_pixels_seed', type=int, default=None, help='Seed for fixed pixel permutation (if enabled)')
    parser.add_argument('--modelnet40_fourier_bands', type=int, default=64, help='Number of Fourier bands for ModelNet40 PE')
    parser.add_argument('--modelnet40_max_freq', type=float, default=1120.0, help='Max frequency for ModelNet40 Fourier PE')
    parser.add_argument('--modelnet40_num_points', type=int, default=2048, help='Number of points for ModelNet40')

    # Model configurations
    parser.add_argument('--num_latents', type=int, default=128, help='Number of latents in the Perceiver model') 
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimension of the latents') 
    parser.add_argument('--num_transformer_blocks', type=int, default=4, help='Number of latent transformer blocks (4-8 recommended)')
    parser.add_argument('--num_cross_attend_stages', type=int, default=1, help='Number of cross-attend stages')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate for attention and MLP layers')
    parser.add_argument('--use_learned_pe', action='store_true', help='Use learned positional encoding instead of Fourier')
    parser.add_argument('--save_attention_maps', '--save_attention_map', action='store_true', default=True, help='Save attention maps for visualization')
    parser.add_argument('--save_metrics', action='store_true', help='Save advanced metrics and confusion matrix')
    parser.add_argument('--attention_save_interval', type=int, default=10, help='Interval (in epochs) for automatically saving attention maps')
    parser.add_argument('--no_weight_sharing', action='store_true', help='Disable weight sharing in latent transformer blocks')
    parser.add_argument('--output_pooling', type=str, default='mean', choices=['mean', 'cls'], help='Method to pool latents for classification: mean or cls (CLS token)')

    # Training configurations
    parser.add_argument('--optimizer', type=str, default='lamb', choices=['lamb', 'adam', 'sgd'], help='Optimizer to use')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--batch_size_cifar10', type=int, default=64, help='Batch size for CIFAR-10') 
    parser.add_argument('--batch_size_modelnet40', type=int, default=32, help='Batch size for ModelNet40 (64-128)') 
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs (100-150)')
    parser.add_argument('--scheduler', type=str, default='multistep', choices=['cosine', 'step', 'multistep', 'none'], help='Learning rate scheduler')
    parser.add_argument('--lr_step_size', type=int, default=30, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for StepLR scheduler')
    parser.add_argument('--eta_min_cosine', type=float, default=0.0, help='Minimum LR for Cosine scheduler')

    # Logging configurations
    parser.add_argument('--log_dir', type=str, default='./logs', help='Directory for logging')
    parser.add_argument('--experiment_name', type=str, default='perceiver_run', help='Name of the experiment for logging')
    parser.add_argument('--use_tensorboard', action='store_true', help='Use TensorBoard for logging')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project_name', type=str, default='perceiver-project', help='WandB project name')

    # System configurations
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')

    return parser

if __name__ == '__main__':
    parser = get_base_config()
    args = parser.parse_args([])
    print("Base config loaded with defaults:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
