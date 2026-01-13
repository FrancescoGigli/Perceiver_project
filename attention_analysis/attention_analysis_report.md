# 🔍 Attention Evolution Analysis Report
Generated: 2025-09-08T13:32:29

## 📊 Overview
- **Total Experiments**: 10
- **Experiment Categories**:
  - Fourier: 2 experiments
  - Learned: 2 experiments
  - RGB-only: 1 experiments
  - Unknown: 3 experiments
  - No Weight Sharing: 1 experiments
  - Fourier+Perm: 1 experiments

## 📋 Experiment Details
### exp1_baseline_fourier
- **PE Type**: Fourier
- **Epochs Available**: 6 (1-51)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 5.8536
  - Sparsity: 0.9876
  - Peak Ratio: 0.0500

### exp2_learned_pe_permuted
- **PE Type**: Learned
- **Epochs Available**: 10 (1-91)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 7.7606
  - Sparsity: 0.9922
  - Peak Ratio: 0.0500

### exp3A_fourier_control
- **PE Type**: Fourier
- **Epochs Available**: 6 (1-51)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 6.3740
  - Sparsity: 0.9872
  - Peak Ratio: 0.0500

### exp3B_rgb_only
- **PE Type**: RGB-only
- **Epochs Available**: 12 (1-111)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 9.0731
  - Sparsity: 0.9997
  - Peak Ratio: 0.0500

### exp4A_weight_sharing_control
- **PE Type**: Unknown
- **Epochs Available**: 5 (1-41)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 6.0667
  - Sparsity: 0.9869
  - Peak Ratio: 0.0500

### exp4B_no_weight_sharing
- **PE Type**: No Weight Sharing
- **Epochs Available**: 8 (1-71)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 6.3807
  - Sparsity: 0.9871
  - Peak Ratio: 0.0500

### exp6_fourier_permuted
- **PE Type**: Fourier+Perm
- **Epochs Available**: 12 (1-111)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 7.4484
  - Sparsity: 0.9912
  - Peak Ratio: 0.0500

### perceiver_cifar10_fourier
- **PE Type**: Unknown
- **Epochs Available**: 12 (1-111)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 6.8913
  - Sparsity: 0.9889
  - Peak Ratio: 0.0500

### perceiver_cifar10_permuted_learned_pe
- **PE Type**: Learned
- **Epochs Available**: 11 (1-101)
- **Final Epoch Stats**:
  - Mean Attention: 0.0039
  - Entropy: 7.0310
  - Sparsity: 0.9892
  - Peak Ratio: 0.0500

### perceiver_modelnet40_fourier
- **PE Type**: Unknown
- **Epochs Available**: 5 (1-41)
- **Final Epoch Stats**:
  - Mean Attention: 0.0005
  - Entropy: 11.9098
  - Sparsity: 1.0000
  - Peak Ratio: 0.0500

## 🔍 Key Findings
Based on the attention pattern analysis:

### Positional Encoding Impact
- **Fourier PE**: Expected to show structured, location-aware patterns
- **Learned PE**: May adapt to data-specific spatial relationships
- **RGB-only**: Likely to show less structured, more content-based patterns

### Robustness Analysis
- **Permuted Data**: Tests spatial invariance of different PE types
- **Weight Sharing**: Impact on attention pattern consistency

### Recommended Next Steps
1. Run comparative analysis across PE types
2. Analyze evolution patterns within each experiment
3. Compare permuted vs non-permuted versions
4. Investigate correlation with model performance