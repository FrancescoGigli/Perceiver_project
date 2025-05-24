# Enhanced visualization of attention maps with comprehensive analysis

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class AttentionVisualizer:
    def __init__(self, colormap='jet', alpha=0.6, figsize=(15, 5)):
        """
        Enhanced attention visualizer with comprehensive analysis features.
        
        Args:
            colormap: Colormap for attention heatmaps
            alpha: Blending factor for overlays
            figsize: Default figure size
        """
        self.colormap = colormap
        self.alpha = alpha
        self.figsize = figsize
        self.available_colormaps = ['jet', 'viridis', 'plasma', 'inferno', 'hot', 'cool', 'seismic']
        
        # Set matplotlib style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    def create_comprehensive_attention_analysis(self, attn_weights, img_tensor, title="Attention", save_path=None):
        """
        Create comprehensive attention analysis with multiple views and statistics.
        """
        # Process inputs
        attn_processed = self._process_attention_weights(attn_weights)
        img_processed = self._process_image_tensor(img_tensor)
        
        if attn_processed is None or img_processed is None:
            print(f"❌ Failed to process inputs for {title}")
            return None
        
        H, W = img_processed.shape[:2]
        attn_map = self._create_attention_map(attn_processed, H, W)
        
        if attn_map is None:
            print(f"❌ Failed to create attention map for {title}")
            return None
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[2, 2, 1.5], width_ratios=[1, 1, 1, 1])
        
        # Row 1: Main visualizations
        self._plot_original_image(fig.add_subplot(gs[0, 0]), img_processed)
        self._plot_attention_heatmap(fig.add_subplot(gs[0, 1]), attn_map)
        self._plot_attention_overlay(fig.add_subplot(gs[0, 2]), img_processed, attn_map)
        self._plot_attention_statistics(fig.add_subplot(gs[0, 3]), attn_map)
        
        # Row 2: Analysis views
        self._plot_attention_distribution(fig.add_subplot(gs[1, 0]), attn_map)
        self._plot_attention_peaks(fig.add_subplot(gs[1, 1]), attn_map, img_processed)
        self._plot_spatial_analysis(fig.add_subplot(gs[1, 2]), attn_map)
        self._plot_attention_entropy(fig.add_subplot(gs[1, 3]), attn_map)
        
        # Row 3: Colormap comparisons
        self._plot_colormap_comparison(fig, gs[2, :], img_processed, attn_map)
        
        plt.suptitle(f'Comprehensive Attention Analysis - {title}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✔ Enhanced visualization saved: {save_path}")
            plt.close()
        
        return fig
    
    def create_evolution_visualization(self, attention_maps_by_epoch, img_tensor, save_path=None):
        """Create visualization showing attention evolution across epochs."""
        if not attention_maps_by_epoch:
            print("❌ No attention maps provided for evolution visualization")
            return None
        
        img_processed = self._process_image_tensor(img_tensor)
        if img_processed is None:
            print("❌ Failed to process image tensor")
            return None
        
        epochs = sorted(attention_maps_by_epoch.keys())
        n_epochs = len(epochs)
        
        fig, axes = plt.subplots(3, n_epochs, figsize=(4*n_epochs, 12))
        if n_epochs == 1:
            axes = axes.reshape(3, 1)
        
        for i, epoch in enumerate(epochs):
            attn_map = attention_maps_by_epoch[epoch]
            
            # Row 1: Attention heatmaps
            im1 = axes[0, i].imshow(attn_map, cmap=self.colormap)
            axes[0, i].set_title(f'Epoch {epoch}', fontweight='bold')
            axes[0, i].axis('off')
            plt.colorbar(im1, ax=axes[0, i], fraction=0.046, pad=0.04)
            
            # Row 2: Overlays
            cmap = plt.get_cmap(self.colormap)
            colored = cmap(attn_map)[:,:,:3]
            overlay = (1-self.alpha)*img_processed + self.alpha*colored
            overlay = np.clip(overlay, 0, 1)
            axes[1, i].imshow(overlay)
            axes[1, i].set_title(f'Overlay E{epoch}', fontweight='bold')
            axes[1, i].axis('off')
            
            # Row 3: Statistics
            self._plot_epoch_statistics(axes[2, i], attn_map, epoch)
        
        plt.suptitle('Attention Evolution Across Training Epochs', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✔ Evolution visualization saved: {save_path}")
            plt.close()
        
        return fig
    
    def _plot_original_image(self, ax, img):
        """Plot original image."""
        ax.imshow(img)
        ax.set_title('Original Image', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_attention_heatmap(self, ax, attn_map):
        """Plot attention heatmap."""
        im = ax.imshow(attn_map, cmap=self.colormap)
        ax.set_title(f'Attention Map ({self.colormap})', fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_attention_overlay(self, ax, img, attn_map):
        """Plot attention overlay on original image."""
        cmap = plt.get_cmap(self.colormap)
        colored = cmap(attn_map)[:,:,:3]
        overlay = (1-self.alpha)*img + self.alpha*colored
        overlay = np.clip(overlay, 0, 1)
        ax.imshow(overlay)
        ax.set_title('Attention Overlay', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_attention_statistics(self, ax, attn_map):
        """Plot attention statistics."""
        stats = {
            'Mean': np.mean(attn_map),
            'Std': np.std(attn_map),
            'Min': np.min(attn_map),
            'Max': np.max(attn_map),
            'Entropy': self._calculate_entropy(attn_map),
            'Sparsity': np.sum(attn_map < 0.1) / attn_map.size,
            'Peak Ratio': np.sum(attn_map > np.percentile(attn_map, 95)) / attn_map.size
        }
        
        ax.axis('off')
        y_pos = 0.95
        ax.text(0.05, y_pos, 'Attention Statistics:', fontsize=12, fontweight='bold', transform=ax.transAxes)
        
        for key, value in stats.items():
            y_pos -= 0.12
            ax.text(0.05, y_pos, f'{key}: {value:.4f}', fontsize=10, transform=ax.transAxes)
    
    def _plot_attention_distribution(self, ax, attn_map):
        """Plot attention value distribution."""
        flat_attn = attn_map.flatten()
        
        # Histogram
        ax.hist(flat_attn, bins=50, alpha=0.7, color='skyblue', edgecolor='black', density=True)
        
        # Statistics lines
        mean_val = np.mean(flat_attn)
        median_val = np.median(flat_attn)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')
        
        # Percentiles
        p95 = np.percentile(flat_attn, 95)
        ax.axvline(p95, color='orange', linestyle=':', linewidth=2, label=f'95th: {p95:.3f}')
        
        ax.set_xlabel('Attention Value')
        ax.set_ylabel('Density')
        ax.set_title('Attention Distribution', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_attention_peaks(self, ax, attn_map, img):
        """Plot attention peaks on the original image."""
        ax.imshow(img)
        
        # Find top attention regions
        threshold = np.percentile(attn_map, 95)
        peak_coords = np.where(attn_map >= threshold)
        
        # Plot peak regions with varying sizes based on attention strength
        for y, x in zip(peak_coords[0], peak_coords[1]):
            attention_strength = attn_map[y, x]
            radius = 2 + 4 * (attention_strength / np.max(attn_map))
            circle = patches.Circle((x, y), radius=radius, linewidth=2, 
                                  edgecolor='red', facecolor='none', alpha=0.8)
            ax.add_patch(circle)
        
        ax.set_title(f'Top 5% Attention Peaks ({len(peak_coords[0])} points)', fontweight='bold')
        ax.axis('off')
    
    def _plot_spatial_analysis(self, ax, attn_map):
        """Plot spatial analysis of attention."""
        # Row and column means
        row_means = np.mean(attn_map, axis=1)
        col_means = np.mean(attn_map, axis=0)
        
        # Create subplot
        ax.clear()
        ax2 = ax.twinx()
        
        # Plot row means
        x_rows = np.arange(len(row_means))
        line1 = ax.plot(x_rows, row_means, 'b-', linewidth=2, label='Row Means')
        ax.set_xlabel('Spatial Position')
        ax.set_ylabel('Row-wise Mean Attention', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot column means (normalized to same scale)
        x_cols = np.linspace(0, len(row_means)-1, len(col_means))
        line2 = ax2.plot(x_cols, col_means, 'r-', linewidth=2, label='Column Means')
        ax2.set_ylabel('Column-wise Mean Attention', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        
        ax.set_title('Spatial Attention Analysis', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_attention_entropy(self, ax, attn_map):
        """Plot attention entropy analysis."""
        # Calculate entropy per row
        row_entropy = []
        for i in range(attn_map.shape[0]):
            row = attn_map[i, :]
            row_entropy.append(self._calculate_entropy(row))
        
        ax.plot(row_entropy, 'g-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Row Index')
        ax.set_ylabel('Entropy')
        ax.set_title('Attention Entropy per Row', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add mean entropy line
        mean_entropy = np.mean(row_entropy)
        ax.axhline(mean_entropy, color='red', linestyle='--', 
                  label=f'Mean: {mean_entropy:.3f}')
        ax.legend()
    
    def _plot_colormap_comparison(self, fig, gs, img, attn_map):
        """Plot comparison of different colormaps."""
        cmaps_to_show = self.available_colormaps[:4]
        
        for i, cmap_name in enumerate(cmaps_to_show):
            ax = fig.add_subplot(gs[i])
            cmap_obj = plt.get_cmap(cmap_name)
            colored = cmap_obj(attn_map)[:,:,:3]
            overlay = (1-self.alpha)*img + self.alpha*colored
            overlay = np.clip(overlay, 0, 1)
            ax.imshow(overlay)
            ax.set_title(f'{cmap_name.title()}', fontsize=10, fontweight='bold')
            ax.axis('off')
    
    def _plot_epoch_statistics(self, ax, attn_map, epoch):
        """Plot statistics for a specific epoch."""
        stats = [
            np.mean(attn_map),
            np.std(attn_map),
            self._calculate_entropy(attn_map),
            np.sum(attn_map > np.percentile(attn_map, 95)) / attn_map.size
        ]
        labels = ['Mean', 'Std', 'Entropy', 'Peak Ratio']
        
        bars = ax.bar(labels, stats, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        ax.set_title(f'E{epoch} Stats', fontweight='bold')
        ax.set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, stats):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _calculate_entropy(self, data):
        """Calculate entropy of data array."""
        # Normalize to probabilities
        data_flat = data.flatten()
        data_norm = data_flat / (np.sum(data_flat) + 1e-8)
        # Calculate entropy
        return -np.sum(data_norm * np.log(data_norm + 1e-8))
    
    def _process_attention_weights(self, attn_weights):
        """Process attention weights to handle various formats."""
        try:
            if isinstance(attn_weights, list):
                attn_weights = self._find_first_tensor(attn_weights)
                if attn_weights is None:
                    return None
            
            if not isinstance(attn_weights, torch.Tensor):
                return None
            
            attn_weights = attn_weights.cpu()
            
            # Handle various dimensions
            while len(attn_weights.shape) > 2:
                if attn_weights.shape[0] == 1:
                    attn_weights = attn_weights.squeeze(0)
                else:
                    attn_weights = attn_weights.mean(dim=0)
            
            return attn_weights.numpy()
        
        except Exception as e:
            print(f"❌ Error processing attention weights: {e}")
            return None
    
    def _process_image_tensor(self, img_tensor):
        """Process image tensor to numpy array."""
        try:
            if isinstance(img_tensor, list):
                img_tensor = self._find_first_tensor(img_tensor)
                if img_tensor is None:
                    return None
            
            if not isinstance(img_tensor, torch.Tensor):
                return None
            
            img_tensor = img_tensor.cpu()
            
            # Handle batch dimension
            if len(img_tensor.shape) == 4 and img_tensor.shape[0] == 1:
                img_tensor = img_tensor.squeeze(0)
            
            # Handle different formats
            if len(img_tensor.shape) == 3:
                # Assume [C, H, W] format
                img_np = img_tensor.permute(1, 2, 0).numpy()
            elif len(img_tensor.shape) == 2:
                # Grayscale image
                img_np = img_tensor.numpy()
                img_np = np.stack([img_np] * 3, axis=-1)  # Convert to RGB
            else:
                return None
            
            # Normalize to [0, 1]
            if img_np.max() > 1.0 or img_np.min() < 0.0:
                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
            
            return img_np
        
        except Exception as e:
            print(f"❌ Error processing image tensor: {e}")
            return None
    
    def _find_first_tensor(self, data, max_depth=5, current_depth=0):
        """Recursively find the first tensor in nested structure."""
        if current_depth > max_depth:
            return None
        
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            for item in data:
                result = self._find_first_tensor(item, max_depth, current_depth + 1)
                if result is not None:
                    return result
        return None
    
    def _create_attention_map(self, attn_processed, H, W):
        """Create attention map matching image dimensions."""
        try:
            if attn_processed.ndim == 1:
                # For 1D attention, try to reshape or resize
                if attn_processed.size == H * W:
                    return attn_processed.reshape(H, W)
                else:
                    # Resize using interpolation
                    side = int(np.sqrt(attn_processed.size))
                    if side * side == attn_processed.size:
                        temp_map = attn_processed.reshape(side, side)
                    else:
                        # Pad or truncate to nearest square
                        target_size = side * side
                        if attn_processed.size > target_size:
                            temp_map = attn_processed[:target_size].reshape(side, side)
                        else:
                            padded = np.pad(attn_processed, (0, target_size - attn_processed.size))
                            temp_map = padded.reshape(side, side)
                    
                    # Resize to target dimensions
                    norm_map = (temp_map - temp_map.min()) / (temp_map.max() - temp_map.min() + 1e-8)
                    resized = Image.fromarray((norm_map * 255).astype(np.uint8))
                    resized = resized.resize((W, H), Image.BICUBIC)
                    return np.array(resized) / 255.0
            
            elif attn_processed.ndim == 2:
                if attn_processed.shape == (H, W):
                    return attn_processed
                else:
                    # Resize 2D attention map
                    norm_map = (attn_processed - attn_processed.min()) / (attn_processed.max() - attn_processed.min() + 1e-8)
                    resized = Image.fromarray((norm_map * 255).astype(np.uint8))
                    resized = resized.resize((W, H), Image.BICUBIC)
                    return np.array(resized) / 255.0
            
            return None
        
        except Exception as e:
            print(f"❌ Error creating attention map: {e}")
            return None

def find_attention_dirs(root_logs):
    """Walk root_logs and yield every path named 'attention_maps'."""
    for dirpath, dirnames, _ in os.walk(root_logs):
        if os.path.basename(dirpath) == 'attention_maps':
            yield dirpath

def main():
    parser = argparse.ArgumentParser(description="Enhanced attention visualization with comprehensive analysis")
    parser.add_argument('--logs_dir', type=str, default='logs', 
                       help='Root logs directory')
    parser.add_argument('--colormap', type=str, default='jet',
                       choices=['jet', 'viridis', 'plasma', 'inferno', 'hot', 'cool', 'seismic'],
                       help='Colormap for attention visualization')
    parser.add_argument('--alpha', type=float, default=0.6,
                       help='Alpha blending factor for overlays (0.0-1.0)')
    parser.add_argument('--create_evolution', action='store_true',
                       help='Create evolution visualization across epochs')
    parser.add_argument('--epochs', nargs='+', type=int, default=None,
                       help='Specific epochs to process (default: all)')
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = AttentionVisualizer(colormap=args.colormap, alpha=args.alpha)
    
    # Find logs directory
    candidates = [args.logs_dir, os.path.join('perceiver_project', args.logs_dir)]
    root_logs = None
    for c in candidates:
        if os.path.isdir(c):
            root_logs = c
            break
    
    if root_logs is None:
        print(f"❌ Cannot find logs directory. Checked: {candidates}")
        return
    
    print(f"📂 Using logs directory: {root_logs}")
    
    # Process all attention_maps directories
    any_dir = False
    for attn_dir in find_attention_dirs(root_logs):
        any_dir = True
        print(f"\n▶ Processing: {attn_dir}")
        
        files = os.listdir(attn_dir)
        pat = re.compile(r'epoch_(\d+)_cross_attn_weights\.pt')
        epochs = sorted(int(m.group(1)) for f in files if (m := pat.match(f)))
        
        if not epochs:
            print("  ⚠️ No cross-attention files found, skipping.")
            continue
        
        # Filter epochs if specified
        if args.epochs:
            epochs = [e for e in epochs if e in args.epochs]
            if not epochs:
                print(f"  ⚠️ None of the specified epochs {args.epochs} found, skipping.")
                continue
        
        print(f"  📊 Found epochs: {epochs}")
        
        # For evolution visualization, collect attention maps
        attention_evolution = {} if args.create_evolution else None
        
        for epoch in epochs:
            cross_file = f'epoch_{epoch}_cross_attn_weights.pt'
            orig_file = f'epoch_{epoch}_original_image_tensor.pt'
            cross_path = os.path.join(attn_dir, cross_file)
            orig_path = os.path.join(attn_dir, orig_file)
            
            if not (os.path.isfile(cross_path) and os.path.isfile(orig_path)):
                print(f"  ❌ Missing files for epoch {epoch}, skipping")
                continue
            
            try:
                # Load data
                attn_weights = torch.load(cross_path, map_location='cpu')
                img_tensor = torch.load(orig_path, map_location='cpu')
                
                # Create comprehensive visualization
                save_path = os.path.join(attn_dir, f'comprehensive_epoch_{epoch}_analysis.png')
                fig = visualizer.create_comprehensive_attention_analysis(
                    attn_weights, img_tensor, 
                    title=f'Epoch {epoch}', save_path=save_path
                )
                
                # Collect for evolution if requested
                if args.create_evolution and fig is not None:
                    attn_processed = visualizer._process_attention_weights(attn_weights)
                    img_processed = visualizer._process_image_tensor(img_tensor)
                    if attn_processed is not None and img_processed is not None:
                        H, W = img_processed.shape[:2]
                        attn_map = visualizer._create_attention_map(attn_processed, H, W)
                        if attn_map is not None:
                            attention_evolution[epoch] = attn_map
            
            except Exception as e:
                print(f"  ❌ Error processing epoch {epoch}: {e}")
                continue
        
        # Create evolution visualization
        if args.create_evolution and attention_evolution:
            evolution_path = os.path.join(attn_dir, 'attention_evolution_analysis.png')
            fig = visualizer.create_evolution_visualization(
                attention_evolution, img_tensor, save_path=evolution_path
            )
    
    if not any_dir:
        print(f"❌ No attention_maps directories found under {root_logs}")
    else:
        print("\n✅ Enhanced attention visualization complete!")
        print("📋 Generated files:")
        print("   • comprehensive_epoch_X_analysis.png - Detailed analysis per epoch")
        if args.create_evolution:
            print("   • attention_evolution_analysis.png - Evolution across epochs")

if __name__ == '__main__':
    main()
