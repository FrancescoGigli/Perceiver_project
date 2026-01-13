#!/usr/bin/env python3
"""
Analyze attention evolution and patterns across different Perceiver experiments.

This script provides comprehensive analysis of attention maps saved during training,
focusing on:
- Early vs late layers pattern evolution
- Cross-experiment comparisons (Fourier vs Learned vs RGB-only PE)
- Quantitative attention statistics
- Qualitative pattern recognition
"""

import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')

class AttentionEvolutionAnalyzer:
    def __init__(self, logs_dir: str = 'logs'):
        """
        Initialize the attention evolution analyzer.
        
        Args:
            logs_dir: Root directory containing experiment logs
        """
        self.logs_dir = Path(logs_dir)
        self.experiments = {}
        self.attention_stats = {}
        
        # Define experiment categories for analysis
        self.experiment_categories = {
            'baseline_fourier': {'pattern': 'baseline|fourier.*control', 'pe_type': 'Fourier', 'color': 'blue'},
            'learned_pe': {'pattern': 'learned.*pe', 'pe_type': 'Learned', 'color': 'green'}, 
            'rgb_only': {'pattern': 'rgb.*only', 'pe_type': 'RGB-only', 'color': 'red'},
            'permuted_fourier': {'pattern': 'fourier.*permut|permut.*fourier', 'pe_type': 'Fourier+Perm', 'color': 'orange'},
            'permuted_learned': {'pattern': 'learned.*permut|permut.*learned', 'pe_type': 'Learned+Perm', 'color': 'purple'},
            'no_weight_sharing': {'pattern': 'no.*weight.*sharing', 'pe_type': 'No Weight Sharing', 'color': 'brown'}
        }
    
    def discover_experiments(self) -> Dict[str, Dict]:
        """
        Discover all available experiments with attention maps.
        
        Returns:
            Dictionary mapping experiment names to their details
        """
        experiments = {}
        
        if not self.logs_dir.exists():
            print(f"❌ Logs directory not found: {self.logs_dir}")
            return experiments
        
        # Search for experiment directories with attention_maps
        for exp_dir in self.logs_dir.iterdir():
            if not exp_dir.is_dir():
                continue
                
            attention_dir = exp_dir / "attention_maps"
            if not attention_dir.exists():
                continue
                
            # Find attention map files
            attn_files = list(attention_dir.glob("epoch_*_cross_attn_weights.pt"))
            if not attn_files:
                continue
                
            # Extract epoch numbers
            epochs = []
            for file in attn_files:
                match = re.search(r'epoch_(\d+)_', file.name)
                if match:
                    epochs.append(int(match.group(1)))
            
            if epochs:
                experiments[exp_dir.name] = {
                    'path': exp_dir,
                    'attention_dir': attention_dir,
                    'epochs': sorted(epochs),
                    'category': self._categorize_experiment(exp_dir.name)
                }
        
        self.experiments = experiments
        print(f"📊 Found {len(experiments)} experiments with attention maps:")
        for exp_name, exp_info in experiments.items():
            category = exp_info['category']
            pe_type = self.experiment_categories[category]['pe_type'] if category else 'Unknown'
            print(f"  • {exp_name}: {len(exp_info['epochs'])} epochs ({pe_type})")
        
        return experiments
    
    def _categorize_experiment(self, exp_name: str) -> Optional[str]:
        """
        Categorize experiment based on its name.
        
        Args:
            exp_name: Experiment name
            
        Returns:
            Category key or None if no match
        """
        exp_name_lower = exp_name.lower()
        
        for category, info in self.experiment_categories.items():
            if re.search(info['pattern'], exp_name_lower):
                return category
        return None
    
    def load_attention_maps(self, exp_name: str, epoch: int) -> Optional[Dict]:
        """
        Load attention maps for a specific experiment and epoch.
        
        Args:
            exp_name: Experiment name
            epoch: Epoch number
            
        Returns:
            Dictionary containing loaded attention data
        """
        if exp_name not in self.experiments:
            return None
            
        exp_info = self.experiments[exp_name]
        attention_dir = exp_info['attention_dir']
        
        # Load files
        attn_file = attention_dir / f"epoch_{epoch}_cross_attn_weights.pt"
        img_file = attention_dir / f"epoch_{epoch}_original_image_tensor.pt"
        processed_file = attention_dir / f"epoch_{epoch}_processed_input.pt"
        
        data = {}
        
        try:
            if attn_file.exists():
                data['attention'] = torch.load(attn_file, map_location='cpu')
            if img_file.exists():
                data['original_image'] = torch.load(img_file, map_location='cpu')
            if processed_file.exists():
                data['processed_input'] = torch.load(processed_file, map_location='cpu')
                
            data['epoch'] = epoch
            data['experiment'] = exp_name
            
            return data if 'attention' in data else None
            
        except Exception as e:
            print(f"❌ Error loading attention maps for {exp_name} epoch {epoch}: {e}")
            return None
    
    def analyze_attention_patterns(self, attention_data: Dict) -> Dict:
        """
        Analyze attention patterns and extract statistics.
        
        Args:
            attention_data: Dictionary containing attention maps
            
        Returns:
            Dictionary with analysis results
        """
        attention_maps = attention_data['attention']
        
        # Handle different attention map formats
        if isinstance(attention_maps, list):
            # Multiple stages/layers
            attention_maps = [self._process_attention_tensor(attn) for attn in attention_maps]
        else:
            # Single attention map
            attention_maps = [self._process_attention_tensor(attention_maps)]
        
        # Calculate statistics for each map
        stats = []
        for i, attn_map in enumerate(attention_maps):
            if attn_map is not None:
                stats.append({
                    'layer': i,
                    'mean': float(attn_map.mean()),
                    'std': float(attn_map.std()),
                    'min': float(attn_map.min()),
                    'max': float(attn_map.max()),
                    'entropy': self._calculate_entropy(attn_map),
                    'sparsity': float((attn_map < 0.1).sum() / attn_map.numel()),
                    'peak_ratio': float((attn_map > attn_map.quantile(0.95)).sum() / attn_map.numel()),
                    'shape': list(attn_map.shape)
                })
        
        return {
            'experiment': attention_data['experiment'],
            'epoch': attention_data['epoch'],
            'num_layers': len(stats),
            'layer_stats': stats,
            'attention_maps': attention_maps
        }
    
    def _process_attention_tensor(self, attn: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Process attention tensor to 2D format for analysis.
        
        Args:
            attn: Raw attention tensor
            
        Returns:
            Processed 2D attention tensor
        """
        try:
            # Remove batch dimension if present
            while len(attn.shape) > 2 and attn.shape[0] == 1:
                attn = attn.squeeze(0)
            
            # Handle different dimensionalities
            if len(attn.shape) == 4:  # [batch, heads, latents, input]
                attn = attn.mean(dim=(0, 1))  # Average over batch and heads
            elif len(attn.shape) == 3:  # [heads, latents, input] or [batch, latents, input]
                attn = attn.mean(dim=0)  # Average over first dimension
            elif len(attn.shape) == 2:  # [latents, input]
                pass  # Already in correct format
            elif len(attn.shape) == 1:  # Flattened
                # Try to reshape to square
                size = int(np.sqrt(len(attn)))
                if size * size == len(attn):
                    attn = attn.reshape(size, size)
                else:
                    return None
            else:
                return None
            
            return attn.float()
            
        except Exception as e:
            print(f"❌ Error processing attention tensor: {e}")
            return None
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """
        Calculate entropy of attention tensor.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Entropy value
        """
        # Normalize to probabilities
        flat = tensor.flatten()
        probs = flat / (flat.sum() + 1e-8)
        
        # Calculate entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        return float(entropy)
    
    def create_evolution_visualization(self, exp_name: str, save_path: Optional[str] = None) -> None:
        """
        Create visualization showing attention evolution across epochs.
        
        Args:
            exp_name: Experiment name
            save_path: Path to save the visualization
        """
        if exp_name not in self.experiments:
            print(f"❌ Experiment {exp_name} not found")
            return
        
        exp_info = self.experiments[exp_name]
        epochs = exp_info['epochs']
        
        # Load attention data for all epochs
        evolution_data = []
        for epoch in epochs:
            data = self.load_attention_maps(exp_name, epoch)
            if data:
                stats = self.analyze_attention_patterns(data)
                evolution_data.append(stats)
        
        if not evolution_data:
            print(f"❌ No attention data found for {exp_name}")
            return
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Attention Evolution - {exp_name}', fontsize=16, fontweight='bold')
        
        # Extract metrics over epochs
        epochs = [data['epoch'] for data in evolution_data]
        means = [np.mean([layer['mean'] for layer in data['layer_stats']]) for data in evolution_data]
        stds = [np.mean([layer['std'] for layer in data['layer_stats']]) for data in evolution_data]
        entropies = [np.mean([layer['entropy'] for layer in data['layer_stats']]) for data in evolution_data]
        sparsities = [np.mean([layer['sparsity'] for layer in data['layer_stats']]) for data in evolution_data]
        peak_ratios = [np.mean([layer['peak_ratio'] for layer in data['layer_stats']]) for data in evolution_data]
        
        # Plot metrics evolution
        axes[0, 0].plot(epochs, means, 'b-o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Mean Attention Value', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(epochs, stds, 'g-o', linewidth=2, markersize=6)
        axes[0, 1].set_title('Attention Standard Deviation', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].plot(epochs, entropies, 'r-o', linewidth=2, markersize=6)
        axes[0, 2].set_title('Attention Entropy', fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Entropy')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].plot(epochs, sparsities, 'm-o', linewidth=2, markersize=6)
        axes[1, 0].set_title('Attention Sparsity', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Sparsity Ratio')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(epochs, peak_ratios, 'c-o', linewidth=2, markersize=6)
        axes[1, 1].set_title('Peak Attention Ratio', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Peak Ratio (>95th percentile)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Show attention patterns for first and last epochs
        if len(evolution_data) >= 2:
            early_attn = evolution_data[0]['attention_maps'][0]
            late_attn = evolution_data[-1]['attention_maps'][0]
            
            # Display attention patterns
            im = axes[1, 2].imshow(late_attn.numpy(), cmap='jet', aspect='auto')
            axes[1, 2].set_title(f'Attention Pattern (Epoch {epochs[-1]})', fontweight='bold')
            axes[1, 2].set_xlabel('Input Elements')
            axes[1, 2].set_ylabel('Latent Queries')
            plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Evolution visualization saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_comparative_analysis(self, experiments: List[str], save_path: Optional[str] = None) -> None:
        """
        Create comparative analysis across multiple experiments.
        
        Args:
            experiments: List of experiment names to compare
            save_path: Path to save the visualization
        """
        valid_experiments = [exp for exp in experiments if exp in self.experiments]
        
        if len(valid_experiments) < 2:
            print(f"❌ Need at least 2 valid experiments for comparison. Found: {valid_experiments}")
            return
        
        # Collect final epoch data for each experiment
        comparison_data = []
        for exp_name in valid_experiments:
            exp_info = self.experiments[exp_name]
            final_epoch = max(exp_info['epochs'])
            
            data = self.load_attention_maps(exp_name, final_epoch)
            if data:
                stats = self.analyze_attention_patterns(data)
                stats['category'] = exp_info['category']
                comparison_data.append(stats)
        
        if len(comparison_data) < 2:
            print(f"❌ Could not load data for comparison")
            return
        
        # Create comparative visualization
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comparative Analysis: Different PE Types', fontsize=16, fontweight='bold')
        
        # Extract metrics for comparison
        exp_names = [data['experiment'] for data in comparison_data]
        categories = [data['category'] for data in comparison_data]
        
        means = [np.mean([layer['mean'] for layer in data['layer_stats']]) for data in comparison_data]
        entropies = [np.mean([layer['entropy'] for layer in data['layer_stats']]) for data in comparison_data]
        sparsities = [np.mean([layer['sparsity'] for layer in data['layer_stats']]) for data in comparison_data]
        peak_ratios = [np.mean([layer['peak_ratio'] for layer in data['layer_stats']]) for data in comparison_data]
        
        # Get colors for each category
        colors = [self.experiment_categories[cat]['color'] if cat else 'gray' for cat in categories]
        pe_types = [self.experiment_categories[cat]['pe_type'] if cat else 'Unknown' for cat in categories]
        
        # Bar plots for comparison
        x_pos = np.arange(len(exp_names))
        
        axes[0, 0].bar(x_pos, means, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Mean Attention Values', fontweight='bold')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(pe_types, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].bar(x_pos, entropies, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Attention Entropy', fontweight='bold')
        axes[0, 1].set_ylabel('Entropy')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(pe_types, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[0, 2].bar(x_pos, sparsities, color=colors, alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Attention Sparsity', fontweight='bold')
        axes[0, 2].set_ylabel('Sparsity Ratio')
        axes[0, 2].set_xticks(x_pos)
        axes[0, 2].set_xticklabels(pe_types, rotation=45, ha='right')
        axes[0, 2].grid(True, alpha=0.3)
        
        axes[1, 0].bar(x_pos, peak_ratios, color=colors, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Peak Attention Ratio', fontweight='bold')
        axes[1, 0].set_ylabel('Peak Ratio')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(pe_types, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Show attention pattern comparison
        if len(comparison_data) >= 2:
            # Display attention patterns from first two experiments
            attn1 = comparison_data[0]['attention_maps'][0]
            attn2 = comparison_data[1]['attention_maps'][0]
            
            # Normalize for better comparison
            attn1_norm = (attn1 - attn1.min()) / (attn1.max() - attn1.min() + 1e-8)
            attn2_norm = (attn2 - attn2.min()) / (attn2.max() - attn2.min() + 1e-8)
            
            im1 = axes[1, 1].imshow(attn1_norm.numpy(), cmap='jet', aspect='auto')
            axes[1, 1].set_title(f'{pe_types[0]} Attention Pattern', fontweight='bold')
            axes[1, 1].set_xlabel('Input Elements')
            axes[1, 1].set_ylabel('Latent Queries')
            plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)
            
            im2 = axes[1, 2].imshow(attn2_norm.numpy(), cmap='jet', aspect='auto')
            axes[1, 2].set_title(f'{pe_types[1]} Attention Pattern', fontweight='bold')
            axes[1, 2].set_xlabel('Input Elements')
            axes[1, 2].set_ylabel('Latent Queries')
            plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Comparative analysis saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_summary_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive summary report.
        
        Args:
            save_path: Path to save the report
            
        Returns:
            Report text
        """
        report_lines = []
        report_lines.append("# 🔍 Attention Evolution Analysis Report")
        report_lines.append(f"Generated: {np.datetime64('now')}")
        report_lines.append("")
        
        # Overview
        report_lines.append("## 📊 Overview")
        report_lines.append(f"- **Total Experiments**: {len(self.experiments)}")
        
        # Categorize experiments
        categories = {}
        for exp_name, exp_info in self.experiments.items():
            category = exp_info['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(exp_name)
        
        report_lines.append("- **Experiment Categories**:")
        for category, exp_list in categories.items():
            if category:
                pe_type = self.experiment_categories[category]['pe_type']
                report_lines.append(f"  - {pe_type}: {len(exp_list)} experiments")
            else:
                report_lines.append(f"  - Unknown: {len(exp_list)} experiments")
        
        report_lines.append("")
        
        # Detailed experiment analysis
        report_lines.append("## 📋 Experiment Details")
        
        for exp_name, exp_info in self.experiments.items():
            category = exp_info['category']
            pe_type = self.experiment_categories[category]['pe_type'] if category else 'Unknown'
            
            report_lines.append(f"### {exp_name}")
            report_lines.append(f"- **PE Type**: {pe_type}")
            report_lines.append(f"- **Epochs Available**: {len(exp_info['epochs'])} ({min(exp_info['epochs'])}-{max(exp_info['epochs'])})")
            
            # Load final epoch statistics
            final_epoch = max(exp_info['epochs'])
            data = self.load_attention_maps(exp_name, final_epoch)
            if data:
                stats = self.analyze_attention_patterns(data)
                avg_stats = {
                    'mean': np.mean([layer['mean'] for layer in stats['layer_stats']]),
                    'entropy': np.mean([layer['entropy'] for layer in stats['layer_stats']]),
                    'sparsity': np.mean([layer['sparsity'] for layer in stats['layer_stats']]),
                    'peak_ratio': np.mean([layer['peak_ratio'] for layer in stats['layer_stats']])
                }
                
                report_lines.append(f"- **Final Epoch Stats**:")
                report_lines.append(f"  - Mean Attention: {avg_stats['mean']:.4f}")
                report_lines.append(f"  - Entropy: {avg_stats['entropy']:.4f}")
                report_lines.append(f"  - Sparsity: {avg_stats['sparsity']:.4f}")
                report_lines.append(f"  - Peak Ratio: {avg_stats['peak_ratio']:.4f}")
            
            report_lines.append("")
        
        # Key findings
        report_lines.append("## 🔍 Key Findings")
        report_lines.append("Based on the attention pattern analysis:")
        report_lines.append("")
        report_lines.append("### Positional Encoding Impact")
        report_lines.append("- **Fourier PE**: Expected to show structured, location-aware patterns")
        report_lines.append("- **Learned PE**: May adapt to data-specific spatial relationships")
        report_lines.append("- **RGB-only**: Likely to show less structured, more content-based patterns")
        report_lines.append("")
        report_lines.append("### Robustness Analysis")
        report_lines.append("- **Permuted Data**: Tests spatial invariance of different PE types")
        report_lines.append("- **Weight Sharing**: Impact on attention pattern consistency")
        report_lines.append("")
        report_lines.append("### Recommended Next Steps")
        report_lines.append("1. Run comparative analysis across PE types")
        report_lines.append("2. Analyze evolution patterns within each experiment")
        report_lines.append("3. Compare permuted vs non-permuted versions")
        report_lines.append("4. Investigate correlation with model performance")
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"✅ Summary report saved: {save_path}")
        
        return report_text


def main():
    parser = argparse.ArgumentParser(description="Analyze attention evolution across Perceiver experiments")
    parser.add_argument('--logs_dir', type=str, default='logs', help='Root logs directory')
    parser.add_argument('--experiments', nargs='+', help='Specific experiments to analyze')
    parser.add_argument('--create_evolution', action='store_true', help='Create evolution visualizations')
    parser.add_argument('--create_comparative_analysis', action='store_true', help='Create comparative analysis')
    parser.add_argument('--save_report', action='store_true', help='Save summary report')
    parser.add_argument('--output_dir', type=str, default='attention_analysis', help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize analyzer
    analyzer = AttentionEvolutionAnalyzer(args.logs_dir)
    
    # Discover experiments
    experiments = analyzer.discover_experiments()
    if not experiments:
        print("❌ No experiments with attention maps found")
        return
    
    # Filter experiments if specified
    if args.experiments:
        available_experiments = [exp for exp in args.experiments if exp in experiments]
        if not available_experiments:
            print(f"❌ None of the specified experiments found: {args.experiments}")
            print(f"Available experiments: {list(experiments.keys())}")
            return
        experiments_to_analyze = available_experiments
    else:
        experiments_to_analyze = list(experiments.keys())
    
    print(f"\n🔍 Analyzing {len(experiments_to_analyze)} experiments:")
    for exp in experiments_to_analyze:
        print(f"  • {exp}")
    
    # Create evolution visualizations
    if args.create_evolution:
        print("\n📈 Creating evolution visualizations...")
        for exp_name in experiments_to_analyze:
            output_path = output_dir / f"{exp_name}_evolution.png"
            analyzer.create_evolution_visualization(exp_name, str(output_path))
    
    # Create comparative analysis
    if args.create_comparative_analysis and len(experiments_to_analyze) >= 2:
        print("\n📊 Creating comparative analysis...")
        output_path = output_dir / "comparative_analysis.png"
        analyzer.create_comparative_analysis(experiments_to_analyze, str(output_path))
    
    # Save summary report
    if args.save_report:
        print("\n📝 Generating summary report...")
        output_path = output_dir / "attention_analysis_report.md"
        analyzer.generate_summary_report(str(output_path))
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
