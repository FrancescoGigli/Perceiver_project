# src/config/base_cfg.py
# Centralized configurations for datasets, model, and training.

import argparse
import torch

def get_base_config():
    parser = argparse.ArgumentParser(description="Base Configuration for Perceiver Project")

    # Dataset configurations
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'modelnet40', 'wikitext2', 'glue_sst2', 'wikitext103',
                                 'glue_cola', 'glue_mrpc', 'glue_stsb', 'glue_qqp',
                                 'glue_mnli', 'glue_qnli', 'glue_rte'],
                        help='Dataset to use')
    parser.add_argument('--data_dir', type=str, default='./data', help='Directory for storing datasets')
    parser.add_argument('--patch_size', type=int, default=1,
                        help='Patch size for CIFAR-10; 1 = raw pixels, as in the paper')
    parser.add_argument('--val_split', type=int, default=5000,
                        help='Images carved out of the 50k training set for validation')
    parser.add_argument('--fourier_num_bands', type=int, default=64,
                        help='Number of Fourier frequency bands K')
    parser.add_argument('--fourier_max_freq', type=float, default=16.0,
                        help='Max Fourier frequency; should equal Nyquist (grid/2)')
    parser.add_argument('--learned_pe_dim', type=int, default=128,
                        help='Dimension of the learned positional encoding')
    parser.add_argument('--permute_pixels', action='store_true',
                        help='Permute the token axis after the PE is generated')
    parser.add_argument('--permute_pixels_seed', type=int, default=42,
                        help='Seed of the fixed token permutation')
    parser.add_argument('--modelnet40_fourier_bands', type=int, default=64, help='Number of Fourier bands for ModelNet40 PE')
    parser.add_argument('--modelnet40_max_freq', type=float, default=1120.0, help='Max frequency for ModelNet40 Fourier PE')
    parser.add_argument('--modelnet40_num_points', type=int, default=2048, help='Number of points for ModelNet40')

    # Model configurations
    parser.add_argument('--model_type', type=str, default='perceiver', choices=['perceiver', 'perceiver_io'],
                        help='Model type to use: perceiver (baseline) or perceiver_io')
    parser.add_argument('--num_latents', type=int, default=128, help='Number of latents in the Perceiver model') 
    parser.add_argument('--latent_dim', type=int, default=512, help='Dimension of the latents') 
    parser.add_argument('--num_transformer_blocks', type=int, default=4, help='Number of latent transformer blocks (4-8 recommended)')
    parser.add_argument('--num_cross_attend_stages', type=int, default=1, help='Number of cross-attend stages')
    parser.add_argument('--num_heads', type=int, default=4, help='Number of attention heads')
    parser.add_argument('--num_heads_cross', type=int, default=1,
                        help='Attention heads in the cross-attention (paper uses 1)')
    parser.add_argument('--num_heads_self', type=int, default=8,
                        help='Attention heads in the latent self-attention')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate for attention and MLP layers')
    parser.add_argument('--use_learned_pe', action='store_true', help='Use learned positional encoding instead of Fourier')
    parser.add_argument('--save_attention_maps', '--save_attention_map', action='store_true', default=False, help='Save attention maps for visualization')
    parser.add_argument('--save_metrics', action='store_true', help='Save advanced metrics and confusion matrix')
    parser.add_argument('--attention_save_interval', type=int, default=10, help='Interval (in epochs) for automatically saving attention maps')
    parser.add_argument('--no_weight_sharing', action='store_true', help='Disable weight sharing in latent transformer blocks')
    parser.add_argument('--cross_attend_arrangement', type=str, default='interleaved',
                        choices=['interleaved', 'at_start'],
                        help='Where the cross-attends sit (Tab. 6)')
    parser.add_argument('--no_latent_transformer', action='store_true',
                        help='Remove the latent transformer entirely (Tab. 5)')
    parser.add_argument('--no_share_cross_attend', action='store_true',
                        help='Give every cross-attend its own weights (required by Tab. 5)')
    parser.add_argument('--latent_init_scale', type=float, default=0.02,
                        help='Std of the truncated normal used for the latent array (Fig. 6)')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Max gradient norm; 0 disables. Prevents LAMB nan divergence at dropout 0')
    parser.add_argument('--use_rotation', action='store_true',
                        help='ModelNet40: random point-cloud rotation')
    parser.add_argument('--use_translation', action='store_true',
                        help='ModelNet40: random per-point translation')
    parser.add_argument('--output_pooling', type=str, default='mean', choices=['mean', 'cls'], help='Method to pool latents for classification: mean or cls (CLS token)')
    parser.add_argument('--no_positional_encoding', action='store_true', help='Disable positional encoding, use only RGB patches')
    parser.add_argument('--num_output_queries', type=int, default=1, help='Number of output queries for Perceiver-IO')
    parser.add_argument('--model_task', type=str, default='classification', choices=['classification', 'mlm'],
                        help='Task for the model: classification or mlm')
    parser.add_argument('--mlm_vocab_size', type=int, default=256, help='Vocabulary size for MLM (byte-level uses 256)')
    parser.add_argument('--text_seq_len', type=int, default=2048, help='Sequence length for byte-level MLM')
    parser.add_argument('--mlm_mask_prob', type=float, default=0.15, help='Mask probability for MLM')
    parser.add_argument('--text_fourier_dim', type=int, default=64, help='Fourier dim for text positional encoding')
    parser.add_argument('--text_max_freq', type=float, default=64.0, help='Max frequency for text positional encoding')
    parser.add_argument('--wikitext2_zip_path', type=str, default=None, help='Optional path to local wikitext-2-v1.zip')
    parser.add_argument('--wikitext103_zip_path', type=str, default=None, help='Optional path to local wikitext-103-v1.zip')

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
    parser.add_argument('--load_checkpoint_path', type=str, default=None, help='Path to a checkpoint to load weights from (for fine-tuning)')

    # System configurations
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (cuda or cpu)')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of dataloader workers')
    parser.add_argument('--no_amp', action='store_true',
                        help='Disable fp16 mixed precision. Full fp32 avoids the LAMB overflow-to-nan divergence')

    return parser

if __name__ == '__main__':
    parser = get_base_config()
    args = parser.parse_args([])
    print("Base config loaded with defaults:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
