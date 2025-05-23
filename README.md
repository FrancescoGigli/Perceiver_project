# Perceiver Experiments

This repository contains implementations and experiments using the Perceiver architecture on the CIFAR-10 dataset. The Perceiver is a neural network architecture that uses attention mechanisms to process inputs of arbitrary structure and size.

## Project Structure

```
perceiver_project/
├── data/               # Data storage directory for datasets
├── logs/               # Training logs and model checkpoints
├── src/
│   ├── config/         # Configuration utilities
│   ├── data/           # Dataset loading and processing
│   ├── perceiver/      # Core Perceiver model implementation
│   └── utils/          # Utility functions and modules
├── train.py            # Main training script
├── visualize_attention.py  # Attention visualization tool
├── visualize_results.py    # Results visualization tool
└── requirements.txt    # Python dependencies
```

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/perceiver_project.git
cd perceiver_project
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Prepare the data directory:
```bash
mkdir -p data
```

## Experiments

This project includes two key experiments using the Perceiver architecture on the CIFAR-10 dataset, focusing on different positional encoding strategies.

### Experiment 1: Fourier Positional Encoding (Baseline)

#### Description
The baseline experiment employs fixed Fourier features as positional encodings. These features are non-learnable and provide a consistent method for encoding positional information across inputs.

#### Configuration

- **Dataset**: CIFAR-10
- **Positional Encoding**: Fixed Fourier features
- **Number of Latents**: 96
- **Latent Dimension**: 384
- **Number of Transformer Blocks**: 4
- **Number of Cross-Attention Stages**: 4
- **Number of Attention Heads**: 3
- **Optimizer**: LAMB
- **Learning Rate**: 0.004
- **Scheduler**: MultiStep
- **Fourier Bands**: 64
- **Maximum Frequency**: 32.0
- **Epochs**: 120
- **Batch Size**: 64
- **Save Attention Maps**: Enabled

#### Command

```bash
python train.py --experiment_name perceiver_cifar10_fourier --dataset cifar10 --data_dir ./data --cifar10_fourier_bands 64 --cifar10_max_freq 32.0 --num_latents 96 --latent_dim 384 --num_cross_attend_stages 4 --num_transformer_blocks 4 --num_heads 3 --dropout 0.2 --output_pooling mean --optimizer lamb --lr 0.004 --scheduler multistep --epochs 120 --batch_size_cifar10 64 --num_workers 4 --save_attention_maps --use_tensorboard
```

### Experiment 2: Learned Positional Encoding

#### Description
This experiment replaces the fixed Fourier features with learnable positional embeddings. The goal is to assess whether the model can benefit from learning positional information directly from the data. Additionally, this experiment includes pixel permutation to test the model's ability to adapt to disrupted spatial patterns.

#### Configuration

- **Dataset**: CIFAR-10
- **Positional Encoding**: Learnable embeddings
- **Pixel Permutation**: Enabled (with seed 42)
- **Number of Latents**: 96
- **Latent Dimension**: 384
- **Number of Transformer Blocks**: 4
- **Number of Cross-Attention Stages**: 1
- **Number of Attention Heads**: 3
- **Optimizer**: LAMB
- **Learning Rate**: 0.004
- **Scheduler**: MultiStep
- **Epochs**: 120
- **Batch Size**: 64
- **Save Attention Maps**: Enabled (every 10 epochs)
- **Save Metrics**: Enabled

#### Command

```bash
python train.py --experiment_name perceiver_cifar10_permuted_learned_pe --dataset cifar10 --data_dir ./data --use_learned_pe --permute_pixels --permute_pixels_seed 42 --num_latents 96 --latent_dim 384 --num_cross_attend_stages 1 --num_transformer_blocks 4 --num_heads 3 --dropout 0.2 --output_pooling mean --optimizer lamb --lr 0.004 --scheduler multistep --epochs 120 --batch_size_cifar10 64 --num_workers 4 --save_attention_maps --attention_save_interval 10 --save_metrics --use_tensorboard
```

### Experiment 3: ModelNet40 3D Point Cloud Classification

#### Description
This experiment demonstrates the Perceiver's capability on 3D point cloud data using the ModelNet40 dataset. The model processes 3D point coordinates using Fourier positional encoding to capture spatial relationships in three-dimensional space.

#### Configuration

- **Dataset**: ModelNet40
- **Data Type**: 3D Point Clouds
- **Number of Points**: 2048 per point cloud
- **Positional Encoding**: Fourier features for 3D coordinates
- **Fourier Bands**: 64
- **Maximum Frequency**: 1120.0
- **Number of Latents**: 128
- **Latent Dimension**: 512
- **Number of Cross-Attention Stages**: 2
- **Number of Transformer Blocks**: 6
- **Number of Attention Heads**: 8
- **Dropout**: 0.1
- **Optimizer**: LAMB
- **Learning Rate**: 0.001
- **Scheduler**: MultiStep
- **Epochs**: 100
- **Batch Size**: 8
- **Save Attention Maps**: Enabled (every 10 epochs)
- **Save Metrics**: Enabled

#### Command

```bash
python train.py --experiment_name perceiver_modelnet40_fourier --dataset modelnet40 --data_dir ./data --modelnet40_num_points 2048 --modelnet40_fourier_bands 64 --modelnet40_max_freq 1120.0 --num_latents 128 --latent_dim 512 --num_cross_attend_stages 2 --num_transformer_blocks 6 --num_heads 8 --dropout 0.1 --output_pooling mean --optimizer lamb --lr 0.001 --scheduler multistep --epochs 120 --batch_size_modelnet40 8 --num_workers 0 --save_attention_maps --attention_save_interval 10 --save_metrics --use_tensorboard
```

## Visualization Tools

The project includes two visualization tools for analyzing model performance and attention patterns.

### Visualize Attention

After training, you can visualize the attention maps to understand how the model attends to different parts of the input:

```bash
python visualize_attention.py
```

This script automatically finds the `attention_maps` directories in your logs and creates overlays of attention patterns on the input images.

### Visualize Results

To visualize training progress and metrics:

```bash
python visualize_results.py --output_dir logs/your_experiment_name --epochs 1 10 20 50 100 --overlay_epoch 100
```

This creates:
1. Training and validation accuracy plots
2. Attention heatmaps for specified epochs
3. An overlay of attention on a sample image for the specified epoch

## Key Parameters Explained

- `--num_latents`: The number of latent vectors in the model (similar to tokens in a transformer)
- `--latent_dim`: Dimensionality of each latent vector
- `--num_cross_attend_stages`: Number of cross-attention iterations between inputs and latents
- `--num_transformer_blocks`: Number of self-attention blocks applied to latents
- `--use_learned_pe`: Enables learned positional encoding instead of Fourier features
- `--permute_pixels`: Randomly permutes image pixels to test spatial invariance
- `--save_attention_maps`: Enables saving of attention maps for visualization
- `--attention_save_interval`: Interval (in epochs) for saving attention maps

## Advanced Usage

### Custom Learning Rate Schedulers

The system supports several learning rate schedulers:

```bash
# Use Cosine Annealing scheduler
python train.py --scheduler cosine --eta_min_cosine 0.00001 [other parameters]

# Use Step scheduler
python train.py --scheduler step --lr_step_size 30 --lr_gamma 0.1 [other parameters]
```

### Weight Sharing

You can control whether to share weights across transformer blocks:

```bash
# Disable weight sharing (use separate weights for each block)
python train.py --no_weight_sharing [other parameters]
```

### Output Pooling Strategies

You can choose between different pooling strategies for the output:

```bash
# Use mean pooling (default)
python train.py --output_pooling mean [other parameters]

# Use CLS token pooling
python train.py --output_pooling cls [other parameters]
