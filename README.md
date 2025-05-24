# 🧠 Perceiver Experiments

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive research project to experiment with the **Perceiver** architecture on different datasets, featuring advanced visualization and analysis tools. The Perceiver is a revolutionary neural architecture that uses attention mechanisms to process inputs of arbitrary structure and size, making it ideal for different modalities like images, audio, video and 3D point clouds.

## 📋 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🏗️ Perceiver Architecture](#️-perceiver-architecture)
- [⚙️ Setup and Installation](#️-setup-and-installation)
- [🧪 Experiments](#-experiments)
- [📊 Advanced Visualization Tools](#-advanced-visualization-tools)
- [📈 Expected Results](#-expected-results)
- [🔧 Key Parameters](#-key-parameters)
- [💡 Advanced Usage](#-advanced-usage)
- [📦 Dependencies](#-dependencies)
- [💡 Tips and Best Practices](#-tips-and-best-practices)
- [❓ FAQ](#-faq)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🚀 Quick Start

Want to get started right away? Follow these steps for a quick setup:

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/perceiver_project.git
cd perceiver_project

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run a test experiment on CIFAR-10
python train.py --experiment_name quick_test --dataset cifar10 --epochs 5 --save_attention_maps

# 5. Visualize results
python visualize_results.py --output_dir logs/quick_test
python visualize_attention.py --logs_dir logs/quick_test
```

## 🏗️ Perceiver Architecture

The **Perceiver** represents an innovative paradigm in multimodal data processing, solving the fundamental scalability problem of traditional Transformers.

### 🔑 Key Concepts

- **Latent Array**: A fixed set of latent vectors that act as the model's "memory"
- **Cross-Attention**: Mechanism that allows latents to "observe" input of any size
- **Self-Attention**: Internal processing of latents through standard transformer layers
- **Positional Encoding**: Position encoding via Fourier features or learnable embeddings

### 🔄 Processing Flow

```
Input (Images/Point Clouds) → Positional Encoding → Cross-Attention → 
Latent Array → Self-Attention Blocks → Output Classification
```

### 💪 Advantages

- **Scalability**: Linear complexity with respect to input size
- **Flexibility**: Handles inputs of different sizes and modalities
- **Efficiency**: Reduces computational cost for large inputs
- **Generalization**: Unified architecture for different data types

### 🧮 Complexity Analysis

**Traditional Transformer**: `O(N²)` where N is the number of input tokens

**Perceiver**: `O(M×N + M²)` where M << N
- M = number of latents (fixed, ~100-500)
- N = input size (variable, thousands-millions)

For a 32×32 image:
- **Standard**: O(1024²) = ~1M operations
- **Perceiver**: O(128×1024 + 128²) = ~144K operations (7x more efficient!)

## ⚙️ Setup and Installation

### 📋 Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: Optional, for GPU acceleration
- **RAM**: Minimum 8GB, recommended 16GB
- **Storage**: At least 5GB of free space

### 🛠️ Detailed Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/perceiver_project.git
cd perceiver_project
```

2. **Create virtual environment**:
```bash
# With venv
python -m venv perceiver_env
source perceiver_env/bin/activate  # Linux/Mac
# perceiver_env\Scripts\activate   # Windows

# Or with conda
conda create -n perceiver_env python=3.9
conda activate perceiver_env
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt

# For advanced features (optional)
pip install plotly scipy scikit-learn tensorboard
```

4. **Verify installation**:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import matplotlib; print('Matplotlib OK')"
```

5. **Prepare directories**:
```bash
mkdir -p data logs
```

## 🧪 Experiments

The project includes three complete experiments to explore different Perceiver capabilities:

### 🎯 Experiment 1: Baseline with Fourier Features

**Objective**: Establish a solid baseline using fixed positional encoding.

#### 📊 Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| Dataset | CIFAR-10 | 32×32 color images, 10 classes |
| Latents | 96 | Number of latent vectors |
| Latent Dim | 384 | Dimensionality of each latent |
| Cross-Attention | 4 stages | Input→latents iterations |
| Transformer Blocks | 4 | Self-attention layers |
| Attention Heads | 3 | Parallel attention heads |
| Fourier Bands | 64 | Frequencies for PE |
| Learning Rate | 0.004 | Learning rate |
| Epochs | 120 | Training epochs |

#### 🚀 Execution Command
```bash
python train.py \
    --experiment_name perceiver_cifar10_fourier \
    --dataset cifar10 \
    --data_dir ./data \
    --cifar10_fourier_bands 64 \
    --cifar10_max_freq 32.0 \
    --num_latents 96 \
    --latent_dim 384 \
    --num_cross_attend_stages 4 \
    --num_transformer_blocks 4 \
    --num_heads 3 \
    --dropout 0.2 \
    --output_pooling mean \
    --optimizer lamb \
    --lr 0.004 \
    --scheduler multistep \
    --epochs 120 \
    --batch_size_cifar10 64 \
    --num_workers 4 \
    --save_attention_maps \
    --use_tensorboard
```

#### 🎯 Expected Results
- **Accuracy**: 85-90% on test set
- **Convergence**: Around epoch 80-100
- **Attention Patterns**: Focus on main objects

---

### 🧠 Experiment 2: Learnable Positional Encoding

**Objective**: Evaluate the impact of learning positional encoding.

#### 📊 Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| PE Type | Learnable | Learned positional embeddings |
| Pixel Permutation | ✅ Seed 42 | Spatial robustness test |
| Cross-Attention | 1 stage | Reduced to compensate complexity |
| Attention Save | Every 10 epochs | For evolution analysis |

#### 🚀 Execution Command
```bash
python train.py \
    --experiment_name perceiver_cifar10_permuted_learned_pe \
    --dataset cifar10 \
    --data_dir ./data \
    --use_learned_pe \
    --permute_pixels \
    --permute_pixels_seed 42 \
    --num_latents 96 \
    --latent_dim 384 \
    --num_cross_attend_stages 1 \
    --num_transformer_blocks 4 \
    --num_heads 3 \
    --dropout 0.2 \
    --output_pooling mean \
    --optimizer lamb \
    --lr 0.004 \
    --scheduler multistep \
    --epochs 120 \
    --batch_size_cifar10 64 \
    --num_workers 4 \
    --save_attention_maps \
    --attention_save_interval 10 \
    --save_metrics \
    --use_tensorboard
```

#### 🎯 Expected Results
- **Adaptability**: Ability to learn disrupted spatial patterns
- **Performance**: Possible 2-5% improvement
- **Convergence**: Slower initially, better long-term

---

### 🌐 Experiment 3: 3D Point Clouds (ModelNet40)

**Objective**: Demonstrate versatility on unstructured 3D data.

#### 📊 Configuration
| Parameter | Value | Description |
|-----------|-------|-------------|
| Dataset | ModelNet40 | 3D point clouds, 40 categories |
| Points | 2048 | Points per cloud |
| Coordinates | (x,y,z) | 3D coordinates |
| Fourier Bands | 64 | For 3D coordinates |
| Max Frequency | 1120.0 | Adapted for 3D range |
| Latents | 128 | Increased for complexity |
| Latent Dim | 512 | Higher dimensionality |
| Cross-Attention | 2 stages | Efficiency/performance balance |
| Transformer Blocks | 6 | Deep processing |
| Attention Heads | 8 | Greater parallelism |

#### 🚀 Execution Command
```bash
python train.py \
    --experiment_name perceiver_modelnet40_fourier \
    --dataset modelnet40 \
    --data_dir ./data \
    --modelnet40_num_points 2048 \
    --modelnet40_fourier_bands 64 \
    --modelnet40_max_freq 1120.0 \
    --num_latents 128 \
    --latent_dim 512 \
    --num_cross_attend_stages 2 \
    --num_transformer_blocks 6 \
    --num_heads 8 \
    --dropout 0.1 \
    --output_pooling mean \
    --optimizer lamb \
    --lr 0.001 \
    --scheduler multistep \
    --epochs 120 \
    --batch_size_modelnet40 8 \
    --num_workers 0 \
    --save_attention_maps \
    --attention_save_interval 10 \
    --save_metrics \
    --use_tensorboard
```

#### 🎯 Expected Results
- **Accuracy**: 85-92% on ModelNet40
- **Robustness**: Invariance to rotations and translations
- **Attention**: Focus on distinctive geometric features

## 📊 Advanced Visualization Tools

The visualization tools represent the heart of the analysis, offering deep insights into both learning patterns and attention mechanisms.

### 🔍 Attention Maps Visualization

#### 🌟 Main Features

- **🎨 Multi-Colormap**: 7 professional color palettes
- **📈 Statistical Analysis**: Mean, deviation, entropy, peaks
- **🔄 Evolution Tracking**: Changes across epochs
- **🎯 Robustness**: Automatic handling of different tensor formats

#### 💻 Basic Usage
```bash
# Automatic analysis of all available maps
python visualize_attention.py

# Advanced customization
python visualize_attention.py \
    --colormap viridis \
    --alpha 0.7 \
    --create_evolution \
    --logs_dir logs/my_experiment
```

#### 🎨 Available Colormaps

| Colormap | Optimal Use | Characteristics |
|----------|-------------|-----------------|
| `viridis` | General analysis | Perceptually uniform |
| `plasma` | Highlight peaks | High contrast |
| `inferno` | Wide range data | Good for print |
| `jet` | Classic visualization | Maximum contrast |
| `hot` | Heat maps | Temperature intuition |
| `cool` | Detailed analysis | Relaxing cool tones |
| `seismic` | Bipolar data | Red-blue divergent |

#### 📁 Generated Output

- **`comprehensive_epoch_X_analysis.png`**: 
  - 6 complete analysis panels
  - Original image + heatmap + overlay
  - Statistics + distribution + peaks

- **`attention_evolution_analysis.png`**:
  - Side-by-side evolution
  - Comparison between epochs
  - Visible learning trends

### 📈 Training Results Visualization

#### 🎯 Complete Dashboard

The dashboard provides a 360° view of the training process:

##### 🏆 Main Panel (Row 1)
- **Accuracy Curves**: Train vs Test with polynomial trends
- **Loss Curves**: Auto-logarithmic scale, key point annotations
- **Live Statistics**: Real-time metrics

##### 🔬 Advanced Analysis (Row 2)
- **Gap Analysis**: Automatic overfitting/underfitting detection
- **Convergence**: Rolling statistics with confidence bands
- **Performance Metrics**: Stability, generalization, improvement

##### 📊 Summary (Row 3)
- **Summary Table**: Best accuracy, min loss, epochs, gap analysis

#### 💻 Usage
```bash
# Standard analysis
python visualize_results.py --output_dir logs/my_experiment

# Complete analysis with attention
python visualize_results.py \
    --output_dir logs/my_experiment \
    --experiment_name "Main_Experiment" \
    --epochs 1 10 20 30 40 50 \
    --enhanced_attention
```

#### 📊 Calculated Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Stability** | `1 - std(last_10_epochs)` | How stable the training is |
| **Convergence** | `1 - abs(final_loss - prev_loss)` | Degree of convergence |
| **Generalization** | `1 - abs(train_acc - test_acc)` | Generalization capability |
| **Improvement** | `final_acc - initial_acc` | Total improvement |
| **Overfitting Score** | `max(0, train_acc - test_acc)` | Overfitting level |

### 🎮 Interactive Demo

We've added a demo script to explore visualizations interactively:

```bash
# Complete visualization demo
python demo_visualizations.py

# Demo with custom dataset
python demo_visualizations.py --logs_dir logs/my_experiment
```

#### 🌟 Demo Features
- **Real-time Visualization**: Automatic map updates
- **Interactive Comparisons**: Side-by-side between different epochs
- **Advanced Export**: Multi-format saving (PNG, SVG, PDF)
- **Parametric Analysis**: Dynamic variation of visualization parameters

## 📈 Expected Results

### 🏆 Performance Benchmark

| Experiment | Dataset | Expected Accuracy | Training Time* | GPU Memory** |
|------------|---------|------------------|----------------|--------------|
| Fourier PE | CIFAR-10 | 87-91% | 2-3 hours | 4-6 GB |
| Learned PE | CIFAR-10 | 85-93% | 3-4 hours | 4-6 GB |
| 3D Point Clouds | ModelNet40 | 88-92% | 4-6 hours | 6-8 GB |

*\*With GPU RTX 3080/4070*  
*\*\*Optimal batch size*

### 📊 Typical Convergence

- **Epochs 1-20**: Fast initial learning (40-60% accuracy)
- **Epochs 20-60**: Steady improvement (60-80% accuracy)  
- **Epochs 60-100**: Fine-tuning and stabilization (80-90% accuracy)
- **Epochs 100+**: Final convergence (plateau around maximum)

### 🎯 Expected Attention Patterns

#### CIFAR-10
- **Central objects**: Focus on airplane wings, car body, etc.
- **Distinctive edges**: Contours and edge features
- **Evolution**: From global patterns to specific details

#### ModelNet40  
- **Geometric features**: Vertices, edges, curved surfaces
- **Symmetries**: Recognition of symmetric patterns
- **Invariance**: Robustness to rotations

## 🔧 Key Parameters

### 🧠 Architecture

| Parameter | Description | Typical Range | Impact |
|-----------|-------------|---------------|--------|
| `--num_latents` | Number of latent vectors | 32-512 | Model capacity |
| `--latent_dim` | Dimension of each latent | 128-1024 | Expressiveness |
| `--num_cross_attend_stages` | Cross-attention stages | 1-8 | Input processing |
| `--num_transformer_blocks` | Self-attention blocks | 2-12 | Processing depth |
| `--num_heads` | Attention heads | 1-16 | Parallelism |

### 📊 Training

| Parameter | Description | Recommended Values | Notes |
|-----------|-------------|-------------------|-------|
| `--optimizer` | Optimization algorithm | `lamb`, `adamw` | LAMB for large batches |
| `--lr` | Learning rate | 0.0001-0.01 | Depends on optimizer |
| `--scheduler` | LR scheduler | `multistep`, `cosine` | For stable convergence |
| `--dropout` | Dropout rate | 0.0-0.3 | Regularization |
| `--batch_size` | Batch size | 16-128 | Limited by GPU memory |

### 🎨 Positional Encoding

| Parameter | Type | Description | When to Use |
|-----------|------|-------------|-------------|
| `--use_learned_pe` | Learned | PE learned by model | Complex spatial pattern data |
| `--fourier_bands` | Fourier | Number of frequency bands | Default for images |
| `--max_freq` | Fourier | Maximum frequency | Adapt to resolution |

### 📈 Monitoring

| Parameter | Description | Recommendation |
|-----------|-------------|----------------|
| `--save_attention_maps` | Save attention maps | Always enable |
| `--attention_save_interval` | Save interval | 10 epochs |
| `--save_metrics` | Save training metrics | Always enable |
| `--use_tensorboard` | TensorBoard logging | For real-time monitoring |

## 💡 Advanced Usage

### 🎛️ Learning Rate Schedulers

#### Cosine Annealing (Recommended)
```bash
python train.py \
    --scheduler cosine \
    --eta_min_cosine 0.00001 \
    --T_max_cosine 100 \
    [other parameters]
```
**Advantages**: Smooth convergence, avoids local plateaus

#### MultiStep (Classic)
```bash
python train.py \
    --scheduler multistep \
    --lr_milestones 30 60 90 \
    --lr_gamma 0.1 \
    [other parameters]
```
**Advantages**: Precise control, boost at specific epochs

#### Step Decay
```bash
python train.py \
    --scheduler step \
    --lr_step_size 30 \
    --lr_gamma 0.5 \
    [other parameters]
```
**Advantages**: Simple, predictable

### ⚙️ Weight Sharing Strategies

#### Shared (Default)
- **Pro**: Fewer parameters, faster training
- **Con**: Limited capacity

#### Non-Shared
```bash
python train.py --no_weight_sharing [other parameters]
```
- **Pro**: Greater expressiveness, better performance
- **Con**: More parameters, slower training

### 🎯 Output Pooling

#### Mean Pooling (Default)
```bash
--output_pooling mean
```
**Use**: General purpose, stable

#### CLS Token
```bash
--output_pooling cls
```
**Use**: When specific global representation is needed

## 📦 Dependencies

### 🛠️ Core Requirements

```txt
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
Pillow>=8.3.0
tqdm>=4.62.0
```

### 📊 Enhanced Visualization

```txt
plotly>=5.0.0          # Interactive plots
scipy>=1.7.0           # Advanced statistical functions
scikit-learn>=1.0.0    # Analysis and preprocessing
tensorboard>=2.7.0     # Training monitoring
```

### 🔬 Research Extensions

```txt
wandb>=0.12.0          # Experiment tracking
optuna>=2.10.0         # Hyperparameter optimization
pytorch-lightning>=1.5.0  # Training framework
torchmetrics>=0.6.0    # Advanced metrics
```

### 📦 Complete Installation

```bash
# Base installation
pip install -r requirements.txt

# Enhanced features
pip install plotly scipy scikit-learn tensorboard

# Research tools
pip install wandb optuna pytorch-lightning torchmetrics

# Development tools
pip install black isort flake8 pytest
```

## 💡 Tips and Best Practices

### 🚀 For Best Results

#### 🎯 Training Strategy
1. **🔥 Warm-up**: Start with low LR for 5-10 epochs
2. **📈 Progressive Scaling**: Gradually increase batch size if possible
3. **🎛️ Hyperparameter Sweep**: Test num_latents in [64, 96, 128, 192]
4. **⏰ Early Stopping**: Monitor validation loss to avoid overfitting
5. **💾 Checkpointing**: Save checkpoints every 20-30 epochs

#### 📊 Monitoring
```bash
# TensorBoard monitoring
tensorboard --logdir logs/ --port 6006

# GPU usage monitoring
watch -n 1 nvidia-smi
```

#### 🎨 Optimal Visualization
1. **📷 Always save**: `--save_attention_maps` always enabled
2. **⏱️ Regular intervals**: `--attention_save_interval 10` for evolution
3. **📊 Complete metrics**: `--save_metrics` for post-training analysis
4. **🔍 Multiple colormaps**: Try different palettes for diverse insights

### 🔧 Troubleshooting

#### ❌ Common Issues

**GPU Out of Memory**
```bash
# Reduce batch size
--batch_size_cifar10 32 --batch_size_modelnet40 4

# Reduce model dimensions
--num_latents 64 --latent_dim 256
```

**Slow Convergence**
```bash
# Increase learning rate
--lr 0.006

# Use more aggressive scheduler
--scheduler cosine --eta_min_cosine 0.0001
```

**Overfitting**
```bash
# Increase dropout
--dropout 0.3

# Reduce model capacity
--num_transformer_blocks 3 --num_latents 48
```

**Underfitting**
```bash
# Increase capacity
--num_latents 128 --latent_dim 512 --num_transformer_blocks 6

# Reduce regularization
--dropout 0.1
```

## ❓ FAQ

### Q: How does the Perceiver differ from standard Transformers?
A: The Perceiver uses a fixed latent bottleneck that processes inputs of any size with O(M×N + M²) complexity instead of O(N²).

### Q: Can I use the Perceiver for my own dataset?
A: Yes! The architecture is modality-agnostic. You'll need to implement a data loader and appropriate positional encoding.

### Q: Why are attention maps important?
A: They reveal what the model focuses on, helping understand decision-making processes and debug potential issues.

### Q: Which positional encoding should I use?
A: Fourier encoding works well for most cases. Use learned PE for complex spatial patterns or when you need adaptation.

### Q: How do I choose the number of latents?
A: Start with 96-128. Increase for complex tasks, decrease if overfitting or memory constraints.

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
