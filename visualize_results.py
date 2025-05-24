# Enhanced visualization of training results and comprehensive model analysis

import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

class ResultsVisualizer:
    def __init__(self, style='seaborn-v0_8', figsize=(12, 8)):
        """
        Enhanced results visualizer with comprehensive analysis features.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        self.style = style
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        # Set style
        try:
            plt.style.use(self.style)
        except:
            print(f"Style {self.style} not available, using default")
            plt.style.use('default')
    
    def create_comprehensive_training_analysis(self, metrics, output_dir, experiment_name=""):
        """Create comprehensive training analysis with multiple visualizations."""
        
        if not metrics:
            print("❌ No metrics provided for analysis")
            return None
        
        # Extract data
        epochs = [m['epoch'] for m in metrics]
        train_acc = [m.get('train_accuracy', 0) for m in metrics]
        test_acc = [m.get('test_accuracy', 0) for m in metrics]
        train_loss = [m.get('train_loss', 0) for m in metrics]
        test_loss = [m.get('test_loss', 0) for m in metrics]
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[2.5, 2.5, 1.5], width_ratios=[2, 2, 1])
        
        # Row 1: Main training curves
        self._plot_accuracy_curves(fig.add_subplot(gs[0, 0]), epochs, train_acc, test_acc, experiment_name)
        self._plot_loss_curves(fig.add_subplot(gs[0, 1]), epochs, train_loss, test_loss, experiment_name)
        self._plot_training_statistics(fig.add_subplot(gs[0, 2]), metrics)
        
        # Row 2: Advanced analysis
        self._plot_learning_curve_analysis(fig.add_subplot(gs[1, 0]), epochs, train_acc, test_acc)
        self._plot_convergence_analysis(fig.add_subplot(gs[1, 1]), epochs, train_acc, test_acc, train_loss, test_loss)
        self._plot_performance_metrics(fig.add_subplot(gs[1, 2]), train_acc, test_acc, train_loss, test_loss)
        
        # Row 3: Summary
        self._plot_training_summary(fig.add_subplot(gs[2, :]), metrics, experiment_name)
        
        plt.suptitle(f'Comprehensive Training Analysis - {experiment_name}', 
                    fontsize=20, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # Save
        save_path = os.path.join(output_dir, f'comprehensive_analysis_{experiment_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✔ Comprehensive analysis saved: {save_path}")
        
        return save_path
    
    def create_attention_heatmap_analysis(self, output_dir, selected_epochs, experiment_name=""):
        """Enhanced attention heatmap visualization."""
        attn_dir = os.path.join(output_dir, 'attention_maps')
        if not os.path.isdir(attn_dir):
            print(f"⚠️ Attention maps directory not found: {attn_dir}")
            return
        
        print(f"📊 Analyzing attention maps for epochs: {selected_epochs}")
        
        for epoch in selected_epochs:
            cross_path = os.path.join(attn_dir, f'epoch_{epoch}_cross_attn_weights.pt')
            
            if not os.path.isfile(cross_path):
                print(f"⚠️ Missing cross-attention for epoch {epoch}, skipping.")
                continue
            
            try:
                # Load attention tensors
                cross_attn = torch.load(cross_path, map_location='cpu')
                
                # Create enhanced heatmap analysis
                fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                
                # Process cross attention
                if isinstance(cross_attn, torch.Tensor):
                    cross_np = cross_attn.detach().cpu().numpy()
                    
                    # Handle different tensor shapes
                    if len(cross_np.shape) >= 3:
                        if len(cross_np.shape) == 4:  # [batch, heads, tgt, src]
                            cross_avg = np.mean(cross_np, axis=(0, 1))
                        else:  # [heads, tgt, src]
                            cross_avg = np.mean(cross_np, axis=0)
                    else:
                        cross_avg = cross_np
                    
                    # Main attention heatmap
                    im1 = axes[0, 0].imshow(cross_avg, cmap='viridis', aspect='auto')
                    axes[0, 0].set_title(f'Cross-Attention Heatmap - Epoch {epoch}', fontweight='bold')
                    plt.colorbar(im1, ax=axes[0, 0])
                    
                    # Attention value distribution
                    axes[0, 1].hist(cross_avg.flatten(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
                    axes[0, 1].set_title('Attention Value Distribution', fontweight='bold')
                    axes[0, 1].set_xlabel('Attention Value')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Top attention regions
                    threshold = np.percentile(cross_avg, 95)
                    binary_map = (cross_avg > threshold).astype(float)
                    axes[0, 2].imshow(binary_map, cmap='Reds', aspect='auto')
                    axes[0, 2].set_title('Top 5% Attention Regions', fontweight='bold')
                    
                    # Row-wise attention analysis
                    row_means = np.mean(cross_avg, axis=1)
                    row_stds = np.std(cross_avg, axis=1)
                    x_rows = np.arange(len(row_means))
                    
                    axes[1, 0].plot(x_rows, row_means, 'b-', linewidth=2, label='Mean')
                    axes[1, 0].fill_between(x_rows, row_means - row_stds, row_means + row_stds, 
                                          alpha=0.3, label='±1 Std')
                    axes[1, 0].set_title('Row-wise Attention Statistics', fontweight='bold')
                    axes[1, 0].set_xlabel('Target Position')
                    axes[1, 0].set_ylabel('Attention Value')
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # Column-wise attention analysis
                    col_means = np.mean(cross_avg, axis=0)
                    col_stds = np.std(cross_avg, axis=0)
                    x_cols = np.arange(len(col_means))
                    
                    axes[1, 1].plot(x_cols, col_means, 'r-', linewidth=2, label='Mean')
                    axes[1, 1].fill_between(x_cols, col_means - col_stds, col_means + col_stds, 
                                          alpha=0.3, label='±1 Std')
                    axes[1, 1].set_title('Column-wise Attention Statistics', fontweight='bold')
                    axes[1, 1].set_xlabel('Source Position')
                    axes[1, 1].set_ylabel('Attention Value')
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    # Attention entropy and sparsity analysis
                    entropy_per_row = []
                    sparsity_per_row = []
                    for i in range(cross_avg.shape[0]):
                        row = cross_avg[i, :]
                        # Entropy
                        row_norm = row / (np.sum(row) + 1e-8)
                        entropy = -np.sum(row_norm * np.log(row_norm + 1e-8))
                        entropy_per_row.append(entropy)
                        # Sparsity (fraction of values below threshold)
                        sparsity = np.sum(row < 0.1) / len(row)
                        sparsity_per_row.append(sparsity)
                    
                    ax_entropy = axes[1, 2]
                    ax_sparsity = ax_entropy.twinx()
                    
                    line1 = ax_entropy.plot(entropy_per_row, 'g-', linewidth=2, label='Entropy')
                    ax_entropy.set_xlabel('Row Index')
                    ax_entropy.set_ylabel('Entropy', color='g')
                    ax_entropy.tick_params(axis='y', labelcolor='g')
                    
                    line2 = ax_sparsity.plot(sparsity_per_row, 'purple', linewidth=2, label='Sparsity')
                    ax_sparsity.set_ylabel('Sparsity', color='purple')
                    ax_sparsity.tick_params(axis='y', labelcolor='purple')
                    
                    ax_entropy.set_title('Entropy & Sparsity Analysis', fontweight='bold')
                    ax_entropy.grid(True, alpha=0.3)
                
                plt.suptitle(f'Enhanced Attention Analysis - {experiment_name} - Epoch {epoch}', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                save_path = os.path.join(output_dir, f'enhanced_attention_epoch_{epoch}_{experiment_name}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"✔ Enhanced attention analysis saved: {save_path}")
                
            except Exception as e:
                print(f"❌ Failed to process epoch {epoch}: {e}")
                continue
    
    def _plot_accuracy_curves(self, ax, epochs, train_acc, test_acc, experiment_name):
        """Plot accuracy curves with advanced features."""
        # Main curves
        ax.plot(epochs, train_acc, 'o-', label='Training Accuracy', 
               color=self.colors[0], linewidth=2.5, markersize=5)
        ax.plot(epochs, test_acc, 's-', label='Test Accuracy', 
               color=self.colors[1], linewidth=2.5, markersize=5)
        
        # Add trend lines if enough data points
        if len(epochs) > 3:
            z_train = np.polyfit(epochs, train_acc, min(2, len(epochs)-1))
            p_train = np.poly1d(z_train)
            z_test = np.polyfit(epochs, test_acc, min(2, len(epochs)-1))
            p_test = np.poly1d(z_test)
            
            ax.plot(epochs, p_train(epochs), '--', alpha=0.7, color=self.colors[0], linewidth=1.5)
            ax.plot(epochs, p_test(epochs), '--', alpha=0.7, color=self.colors[1], linewidth=1.5)
        
        # Highlight best performance
        if test_acc:
            best_test_idx = np.argmax(test_acc)
            ax.annotate(f'Best: {test_acc[best_test_idx]:.4f}@E{epochs[best_test_idx]}',
                       xy=(epochs[best_test_idx], test_acc[best_test_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=10, fontweight='bold')
        
        ax.set_title('Training & Test Accuracy Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    def _plot_loss_curves(self, ax, epochs, train_loss, test_loss, experiment_name):
        """Plot loss curves with logarithmic scale if needed."""
        ax.plot(epochs, train_loss, 'o-', label='Training Loss', 
               color=self.colors[2], linewidth=2.5, markersize=5)
        ax.plot(epochs, test_loss, 's-', label='Test Loss', 
               color=self.colors[3], linewidth=2.5, markersize=5)
        
        # Use log scale if loss values vary greatly
        valid_losses = [x for x in train_loss + test_loss if x > 0]
        if valid_losses and max(valid_losses) / min(valid_losses) > 100:
            ax.set_yscale('log')
        
        # Highlight minimum loss
        if test_loss:
            min_test_idx = np.argmin(test_loss)
            ax.annotate(f'Min Loss: {test_loss[min_test_idx]:.6f}@E{epochs[min_test_idx]}',
                       xy=(epochs[min_test_idx], test_loss[min_test_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgreen', alpha=0.8),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
                       fontsize=10, fontweight='bold')
        
        ax.set_title('Training & Test Loss Evolution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_training_statistics(self, ax, metrics):
        """Plot key training statistics."""
        stats_data = {
            'Final Train Acc': metrics[-1].get('train_accuracy', 0),
            'Final Test Acc': metrics[-1].get('test_accuracy', 0),
            'Best Test Acc': max([m.get('test_accuracy', 0) for m in metrics]),
            'Final Train Loss': metrics[-1].get('train_loss', 0),
            'Final Test Loss': metrics[-1].get('test_loss', 0),
            'Min Test Loss': min([m.get('test_loss', float('inf')) for m in metrics 
                                if m.get('test_loss', float('inf')) != float('inf')], default=0)
        }
        
        # Create text display
        ax.axis('off')
        y_pos = 0.95
        ax.text(0.05, y_pos, 'Training Statistics:', fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        for key, value in stats_data.items():
            y_pos -= 0.12
            if 'Acc' in key:
                ax.text(0.05, y_pos, f'{key}: {value:.4f} ({value*100:.2f}%)', 
                       fontsize=11, transform=ax.transAxes)
            else:
                ax.text(0.05, y_pos, f'{key}: {value:.6f}', 
                       fontsize=11, transform=ax.transAxes)
        
        # Add overfitting indicator
        final_gap = abs(metrics[-1].get('train_accuracy', 0) - metrics[-1].get('test_accuracy', 0))
        y_pos -= 0.15
        color = 'red' if final_gap > 0.1 else 'green' if final_gap < 0.05 else 'orange'
        ax.text(0.05, y_pos, f'Train-Test Gap: {final_gap:.4f}', 
               fontsize=11, color=color, fontweight='bold', transform=ax.transAxes)
        
        # Add training stability indicator
        test_accs = [m.get('test_accuracy', 0) for m in metrics[-5:]]  # Last 5 epochs
        stability = 1 - np.std(test_accs) if len(test_accs) > 1 else 1
        y_pos -= 0.12
        stability_color = 'green' if stability > 0.95 else 'orange' if stability > 0.90 else 'red'
        ax.text(0.05, y_pos, f'Stability: {stability:.4f}', 
               fontsize=11, color=stability_color, fontweight='bold', transform=ax.transAxes)
    
    def _plot_learning_curve_analysis(self, ax, epochs, train_acc, test_acc):
        """Analyze learning curves for overfitting/underfitting."""
        if not train_acc or not test_acc:
            ax.text(0.5, 0.5, 'Insufficient data for gap analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return
        
        gap = np.array(train_acc) - np.array(test_acc)
        
        ax.plot(epochs, gap, 'o-', color='purple', linewidth=2.5, markersize=5)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
        
        # Fill areas
        ax.fill_between(epochs, gap, 0, where=(gap > 0), color='red', alpha=0.3, 
                       label='Overfitting', interpolate=True)
        ax.fill_between(epochs, gap, 0, where=(gap < 0), color='blue', alpha=0.3, 
                       label='Underfitting', interpolate=True)
        
        # Add trend line
        if len(epochs) > 2:
            z = np.polyfit(epochs, gap, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), 'r--', alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_title('Overfitting/Underfitting Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Train Acc - Test Acc')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_convergence_analysis(self, ax, epochs, train_acc, test_acc, train_loss, test_loss):
        """Analyze convergence patterns."""
        if len(epochs) < 5:
            ax.text(0.5, 0.5, 'Not enough data\nfor convergence analysis', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return
        
        # Calculate rolling statistics
        window = min(5, len(epochs) // 3)
        
        def rolling_stat(data, window, stat_func):
            return [stat_func(data[max(0, i-window):i+1]) for i in range(len(data))]
        
        rolling_mean_acc = rolling_stat(test_acc, window, np.mean)
        rolling_std_acc = rolling_stat(test_acc, window, np.std)
        
        ax.plot(epochs, test_acc, 'o-', label='Test Accuracy', alpha=0.6, markersize=4)
        ax.plot(epochs, rolling_mean_acc, '-', linewidth=3, label=f'Rolling Mean (w={window})', color='red')
        ax.fill_between(epochs, 
                       np.array(rolling_mean_acc) - np.array(rolling_std_acc),
                       np.array(rolling_mean_acc) + np.array(rolling_std_acc),
                       alpha=0.3, label='±1 Std', color='red')
        
        ax.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Test Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_performance_metrics(self, ax, train_acc, test_acc, train_loss, test_loss):
        """Plot performance metrics as a bar chart."""
        if not train_acc or not test_acc:
            ax.text(0.5, 0.5, 'Insufficient data\nfor metrics calculation', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            return
        
        # Calculate metrics
        metrics = {
            'Final Accuracy': test_acc[-1],
            'Stability': 1 - np.std(test_acc[-10:]) if len(test_acc) >= 10 else 1 - np.std(test_acc),
            'Convergence': max(0, 1 - abs(test_loss[-1] - test_loss[-5]) if len(test_loss) >= 5 else 1),
            'Generalization': max(0, 1 - abs(train_acc[-1] - test_acc[-1])),
            'Improvement': (test_acc[-1] - test_acc[0]) if len(test_acc) > 1 else 0
        }
        
        labels = list(metrics.keys())
        values = list(metrics.values())
        
        bars = ax.bar(labels, values, color=self.colors[:len(labels)], alpha=0.8)
        ax.set_title('Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_training_summary(self, ax, metrics, experiment_name):
        """Create a comprehensive training summary table."""
        ax.axis('off')
        
        # Calculate summary statistics
        train_accs = [m.get('train_accuracy', 0) for m in metrics]
        test_accs = [m.get('test_accuracy', 0) for m in metrics]
        train_losses = [m.get('train_loss', 0) for m in metrics]
        test_losses = [m.get('test_loss', 0) for m in metrics]
        
        summary_data = [
            ['Metric', 'Value', 'Epoch', 'Details'],
            ['Best Test Accuracy', f'{max(test_accs):.4f}', f'{np.argmax(test_accs) + 1}', f'{max(test_accs)*100:.2f}%'],
            ['Final Test Accuracy', f'{test_accs[-1]:.4f}', f'{len(test_accs)}', f'{test_accs[-1]*100:.2f}%'],
            ['Lowest Test Loss', f'{min(test_losses):.6f}', f'{np.argmin(test_losses) + 1}', 'Best convergence'],
            ['Training Epochs', f'{len(metrics)}', '-', f'{len(metrics)} completed'],
            ['Overfitting Score', f'{max(0, train_accs[-1] - test_accs[-1]):.4f}', '-', 
             'Good' if max(0, train_accs[-1] - test_accs[-1]) < 0.05 else 'Moderate' if max(0, train_accs[-1] - test_accs[-1]) < 0.15 else 'High'],
            ['Accuracy Improvement', f'{test_accs[-1] - test_accs[0]:.4f}', '-', 
             f'+{(test_accs[-1] - test_accs[0])*100:.2f}%' if test_accs[-1] > test_accs[0] else f'{(test_accs[-1] - test_accs[0])*100:.2f}%']
        ]
        
        # Create table
        table = ax.table(cellText=summary_data[1:], colLabels=summary_data[0],
                        cellLoc='center', loc='center',
                        colWidths=[0.3, 0.2, 0.15, 0.35])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)
        
        # Style the table
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header row
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#f2f2f2' if i % 2 == 0 else 'white')
        
        ax.set_title(f'Training Summary - {experiment_name}', fontsize=14, fontweight='bold', pad=20)

def load_training_metrics(metrics_file):
    """Load training metrics from JSON file."""
    try:
        with open(metrics_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Error loading metrics from {metrics_file}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Enhanced results visualization with comprehensive analysis")
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory containing training results')
    parser.add_argument('--experiment_name', type=str, default="",
                       help='Experiment name for labeling')
    parser.add_argument('--epochs', nargs='+', type=int, default=None,
                       help='Specific epochs to analyze for attention (default: all available)')
    parser.add_argument('--enhanced_attention', action='store_true',
                       help='Create enhanced attention analysis')
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        print(f"❌ Output directory not found: {args.output_dir}")
        return
    
    # Initialize visualizer
    visualizer = ResultsVisualizer()
    
    # Set experiment name if not provided
    if not args.experiment_name:
        args.experiment_name = os.path.basename(args.output_dir.rstrip('/'))
    
    print(f"📊 Analyzing experiment: {args.experiment_name}")
    print(f"📂 Output directory: {args.output_dir}")
    
    # Load training metrics
    metrics_file = os.path.join(args.output_dir, 'training_metrics.json')
    if os.path.isfile(metrics_file):
        print(f"📈 Loading training metrics from: {metrics_file}")
        metrics = load_training_metrics(metrics_file)
        
        if metrics:
            # Create comprehensive training analysis
            analysis_path = visualizer.create_comprehensive_training_analysis(
                metrics, args.output_dir, args.experiment_name
            )
            print(f"✔ Training analysis complete: {analysis_path}")
        else:
            print("⚠️ No valid metrics found for training analysis")
    else:
        print(f"⚠️ Training metrics file not found: {metrics_file}")
    
    # Enhanced attention analysis if requested
    if args.enhanced_attention:
        attn_dir = os.path.join(args.output_dir, 'attention_maps')
        if os.path.isdir(attn_dir):
            # Find available epochs
            files = os.listdir(attn_dir)
            available_epochs = sorted(set(
                int(f.split('_')[1]) for f in files 
                if f.startswith('epoch_') and f.endswith('_cross_attn_weights.pt')
            ))
            
            # Use specified epochs or all available
            epochs_to_analyze = args.epochs if args.epochs else available_epochs
            epochs_to_analyze = [e for e in epochs_to_analyze if e in available_epochs]
            
            if epochs_to_analyze:
                print(f"🔍 Creating enhanced attention analysis for epochs: {epochs_to_analyze}")
                visualizer.create_attention_heatmap_analysis(
                    args.output_dir, epochs_to_analyze, args.experiment_name
                )
            else:
                print("⚠️ No valid epochs found for attention analysis")
        else:
            print(f"⚠️ Attention maps directory not found: {attn_dir}")
    
    print("\n✅ Enhanced visualization analysis complete!")
    print("📋 Generated files:")
    print(f"   • comprehensive_analysis_{args.experiment_name}.png - Complete training analysis")
    if args.enhanced_attention:
        print("   • enhanced_attention_epoch_X_[name].png - Detailed attention analysis per epoch")

if __name__ == '__main__':
    main()
