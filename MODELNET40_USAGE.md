# ModelNet40 Augmentation Study - Usage Guide

## 🎯 Overview

This script runs 3 ModelNet40 experiments to study data augmentation effects on Perceiver classification performance, targeting the paper's **85.7% top-1 accuracy**.

## 📋 Experiments

1. **`modelnet40_baseline`** - Scale only (0.99-1.01) - Target: 85.7%
2. **`modelnet40_with_translation`** - Scale + Translation (-0.02 to +0.02)  
3. **`modelnet40_with_rotation`** - Scale + Random Rotation

## 🚀 Usage

### Run All Experiments
```bash
python run_modelnet40_augmentation_study.py
```

### Run Specific Experiment
```bash
python run_modelnet40_augmentation_study.py --experiment modelnet40_baseline
```

### Dry Run (Test Without Training)
```bash
python run_modelnet40_augmentation_study.py --dry-run
```

### Custom Log Directory
```bash
python run_modelnet40_augmentation_study.py --log-dir my_logs
```

## ⚙️ Paper Configuration

- **Dataset**: ModelNet40 (9,843 train / 2,468 test)
- **Input**: ~2048 points (x,y,z) per object
- **Architecture**: 2 cross-attention + 6 self-attention blocks
- **PE**: Fourier features, freq_max=1120, 64 bands
- **Training**: batch_size=512, LAMB optimizer, LR=0.001 (constant)
- **Target**: 85.7% top-1 accuracy

## 📊 Output

The script generates:

1. **Real-time training logs** for each experiment
2. **TensorBoard logs** in `logs/[experiment_name]/`
3. **Final comparison report** with accuracy vs target
4. **JSON report** `modelnet40_augmentation_study_report.json`

## 📈 Expected Results

Based on the paper:
- **Baseline (scale only)**: ~85.7% accuracy
- **+ Translation**: ≤85.7% (paper states no improvement)
- **+ Rotation**: ≤85.7% (paper states no improvement)

## ⚡ Quick Test

Test the setup without training:
```bash
python run_modelnet40_augmentation_study.py --dry-run
```

## 📁 Generated Files

```
logs/
├── modelnet40_baseline/
│   ├── config.txt
│   ├── checkpoints/best_model.pt
│   └── tensorboard_logs/
├── modelnet40_with_translation/
├── modelnet40_with_rotation/
├── modelnet40_baseline_train.py        # Generated training script
├── modelnet40_with_translation_train.py
├── modelnet40_with_rotation_train.py
└── modelnet40_augmentation_study_report.json
```

## ⏱️ Estimated Runtime

- **Single experiment**: ~3-6 hours (depending on hardware)
- **All 3 experiments**: ~9-18 hours total
- **Early stopping**: Included (patience=15 epochs)

## 🔧 Requirements

- CUDA-capable GPU (recommended for batch_size=512)
- PyTorch + torch-geometric
- ModelNet40 dataset (auto-downloaded)
- ~8GB+ GPU memory for batch_size=512

## 🎯 Success Criteria

✅ **Success**: Baseline experiment achieves ~85.7% accuracy
✅ **Paper Confirmation**: Translation/rotation don't improve results
