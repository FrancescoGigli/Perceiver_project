# Perceiver Experiments on CIFAR-10

This section outlines two key experiments conducted using the Perceiver architecture on the CIFAR-10 dataset, focusing on the impact of different positional encoding strategies.

## Experiment 1: Fourier Positional Encoding (Baseline)

### Description:
The baseline experiment employs fixed Fourier features as positional encodings. These features are non-learnable and provide a consistent method for encoding positional information across inputs.

### Configuration:

- **Dataset**: CIFAR-10
- **Positional Encoding**: Fixed Fourier features
- **Number of Latents**: 128
- **Latent Dimension**: 512
- **Number of Transformer Blocks**: 4
- **Number of Cross-Attention Stages**: 1
- **Number of Attention Heads**: 4
- **Optimizer**: LAMB
- **Learning Rate Scheduler**: Cosine
- **Fourier Bands**: 64
- **Maximum Frequency**: 32.0
- **Epochs**: 120
- **Save Attention Maps**: Enabled

### Command:

```cmd
python train.py --experiment_name perceiver_cifar10_fourier --dataset cifar10 --data_dir ./data --cifar10_fourier_bands 64 --cifar10_max_freq 32.0 --num_latents 96 --latent_dim 384 --num_cross_attend_stages 4 --num_transformer_blocks 4 --num_heads 3 --dropout 0.2 --output_pooling mean --optimizer lamb --lr 0.004 --scheduler multistep --epochs 120 --batch_size_cifar10 64 --num_workers 4 --save_attention_maps --use_tensorboard  ```

## Experiment 2: Learned Positional Encoding

### Description:
This experiment replaces the fixed Fourier features with learnable positional embeddings. The goal is to assess whether the model can benefit from learning positional information directly from the data.

### Configuration:

- **Dataset**: CIFAR-10
- **Positional Encoding**: Learnable embeddings
- **Number of Latents**: 128
- **Latent Dimension**: 512
- **Number of Transformer Blocks**: 4
- **Number of Cross-Attention Stages**: 1
- **Number of Attention Heads**: 4
- **Optimizer**: LAMB
- **Learning Rate Scheduler**: Cosine
- **Epochs**: 120
- **Save Attention Maps**: Enabled

### Command:

```cmd
python train.py --experiment_name perceiver_cifar10_permuted_learned_pe --dataset cifar10 --data_dir ./data --use_learned_pe --permute_pixels --permute_pixels_seed 42 --num_latents 96 --latent_dim 384 --num_cross_attend_stages 1 --num_transformer_blocks 4 --num_heads 3 --dropout 0.2 --output_pooling mean --optimizer lamb --lr 0.004 --scheduler multistep --epochs 120 --batch_size_cifar10 64 --num_workers 4 --save_attention_maps --attention_save_interval 10 --save_metrics --use_tensorboard


```
