#!/usr/bin/env python3
"""
Perceiver-specific attention visualization 
Creates visualizations similar to the original Perceiver paper
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PerceiverAttentionVisualizer:
    def __init__(self, figsize=(20, 12)):
        self.figsize = figsize
        plt.style.use('default')
    
    def create_perceiver_style_visualization(self, attn_weights, img_tensor, title="Perceiver Attention", save_path=None):
        """
        Create Perceiver-style visualization showing:
        - Original image
        - Attention overlay  
        - Grid of individual attention maps (checkerboard patterns)
        """
        # Process inputs
        img_processed = self._process_image_tensor(img_tensor)
        if img_processed is None:
            print(f"❌ Failed to process image tensor for {title}")
            return None
            
        # Process attention weights to get multiple maps
        attention_maps = self._extract_multiple_attention_maps(attn_weights)
        if not attention_maps:
            print(f"❌ Failed to extract attention maps for {title}")
            return None
        
        print(f"📊 Processing {len(attention_maps)} attention maps for {title}")
        
        # Create figure with specific layout
        fig = plt.figure(figsize=self.figsize)
        
        # Layout: Top row with original + overlay + stats, bottom with attention grids
        n_maps = len(attention_maps)
        n_cols_grid = min(16, n_maps)  # Max 16 columns for readability
        n_rows_needed = (n_maps + n_cols_grid - 1) // n_cols_grid
        
        # Create subplot layout
        if n_rows_needed == 1:
            # Single row of attention maps
            gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1], hspace=0.3, wspace=0.2)
            
            # Top row
            ax_orig = fig.add_subplot(gs[0, 0])
            ax_overlay = fig.add_subplot(gs[0, 1])
            ax_summary = fig.add_subplot(gs[0, 2])
            ax_stats = fig.add_subplot(gs[0, 3])
            
            # Attention maps grid
            ax_grid = fig.add_subplot(gs[1:, :])
            
        else:
            # Multiple rows needed
            gs = fig.add_gridspec(2 + n_rows_needed, 4, height_ratios=[1, 1] + [1]*n_rows_needed, hspace=0.3, wspace=0.2)
            
            # Top row
            ax_orig = fig.add_subplot(gs[0, 0])
            ax_overlay = fig.add_subplot(gs[0, 1]) 
            ax_summary = fig.add_subplot(gs[0, 2])
            ax_stats = fig.add_subplot(gs[0, 3])
            
            # Second row for averaged attention
            ax_avg = fig.add_subplot(gs[1, :])
            
            # Attention maps grid
            ax_grid = fig.add_subplot(gs[2:, :])
        
        # Plot original image
        self._plot_original_image(ax_orig, img_processed)
        
        # Create and plot overlay with averaged attention
        avg_attention = self._create_averaged_attention_map(attention_maps, img_processed.shape[:2])
        if avg_attention is not None:
            self._plot_attention_overlay(ax_overlay, img_processed, avg_attention)
            self._plot_attention_heatmap(ax_summary, avg_attention, "Average Attention")
            self._plot_attention_statistics(ax_stats, avg_attention)
        
        # Create grid of individual attention maps
        self._plot_attention_grid(ax_grid, attention_maps, img_processed.shape[:2], n_cols_grid)
        
        plt.suptitle(f'{title} - Perceiver Attention Analysis', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Perceiver-style visualization saved: {save_path}")
            
        return fig
    
    def create_layer_comparison(self, attention_data_by_layer, img_tensor, title="Layer Comparison", save_path=None):
        """
        Create comparison of attention across different layers
        Similar to Figure 3 in the Perceiver paper
        """
        img_processed = self._process_image_tensor(img_tensor)
        if img_processed is None:
            return None
            
        layers = sorted(attention_data_by_layer.keys())
        n_layers = len(layers)
        
        fig, axes = plt.subplots(n_layers + 1, 1, figsize=(20, 4 * (n_layers + 1)))
        if n_layers == 0:
            return None
            
        # First row: original image and layer summaries
        axes[0].imshow(img_processed)
        axes[0].set_title('Original Image', fontweight='bold')
        axes[0].axis('off')
        
        # Process each layer
        for i, layer in enumerate(layers):
            attention_maps = self._extract_multiple_attention_maps(attention_data_by_layer[layer])
            if attention_maps:
                self._plot_layer_attention_row(axes[i+1], attention_maps, img_processed.shape[:2], f'Layer {layer}')
        
        plt.suptitle(f'{title} - Cross-Attention Layers', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"✅ Layer comparison saved: {save_path}")
            
        return fig
    
    def _plot_original_image(self, ax, img):
        """Plot original image."""
        ax.imshow(img)
        ax.set_title('Original Image', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_attention_overlay(self, ax, img, attn_map, alpha=0.6):
        """Plot attention overlay on original image."""
        # Create colored overlay
        cmap = plt.get_cmap('jet')
        colored = cmap(attn_map)[:,:,:3]
        overlay = (1-alpha)*img + alpha*colored
        overlay = np.clip(overlay, 0, 1)
        
        ax.imshow(overlay)
        ax.set_title('Attention Overlay', fontsize=12, fontweight='bold')
        ax.axis('off')
    
    def _plot_attention_heatmap(self, ax, attn_map, title="Attention Map"):
        """Plot attention heatmap."""
        im = ax.imshow(attn_map, cmap='jet')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    def _plot_attention_statistics(self, ax, attn_map):
        """Plot attention statistics."""
        stats = {
            'Mean': np.mean(attn_map),
            'Std': np.std(attn_map),
            'Max': np.max(attn_map),
            'Min': np.min(attn_map),
            'Sparsity': np.sum(attn_map < 0.1) / attn_map.size,
            'Peak %': np.sum(attn_map > np.percentile(attn_map, 95)) / attn_map.size * 100
        }
        
        ax.axis('off')
        y_pos = 0.95
        ax.text(0.05, y_pos, 'Statistics:', fontsize=14, fontweight='bold', transform=ax.transAxes)
        
        for key, value in stats.items():
            y_pos -= 0.12
            ax.text(0.05, y_pos, f'{key}: {value:.3f}', fontsize=11, transform=ax.transAxes)
    
    def _plot_attention_grid(self, ax, attention_maps, target_shape, n_cols):
        """Plot grid of attention maps similar to Perceiver paper."""
        n_maps = len(attention_maps)
        n_rows = (n_maps + n_cols - 1) // n_cols
        
        # Create combined image
        H, W = target_shape
        grid_H = H * n_rows
        grid_W = W * n_cols
        combined_image = np.ones((grid_H, grid_W)) * 0.5  # Gray background
        
        for idx, attn_data in enumerate(attention_maps):
            row = idx // n_cols
            col = idx % n_cols
            
            # Resize attention map to target shape
            attn_map = self._resize_attention_map(attn_data, target_shape)
            if attn_map is not None:
                # Place in grid
                start_row = row * H
                end_row = start_row + H
                start_col = col * W
                end_col = start_col + W
                combined_image[start_row:end_row, start_col:end_col] = attn_map
        
        # Display grid
        ax.imshow(combined_image, cmap='gray', aspect='auto')
        ax.set_title(f'Attention Maps Grid ({n_maps} maps)', fontsize=14, fontweight='bold')
        ax.axis('off')
        
        # Add grid lines to separate maps
        for i in range(1, n_rows):
            ax.axhline(y=i*H-0.5, color='white', linewidth=2)
        for j in range(1, n_cols):
            ax.axvline(x=j*W-0.5, color='white', linewidth=2)
    
    def _plot_layer_attention_row(self, ax, attention_maps, target_shape, layer_name):
        """Plot a row of attention maps for a specific layer."""
        n_maps = len(attention_maps)
        if n_maps == 0:
            ax.axis('off')
            return
            
        # Limit number of maps to display (for readability)
        max_display = 20
        maps_to_show = attention_maps[:max_display]
        
        # Create horizontal concatenation
        H, W = target_shape
        combined_width = W * len(maps_to_show)
        combined_image = np.ones((H, combined_width)) * 0.5
        
        for idx, attn_data in enumerate(maps_to_show):
            attn_map = self._resize_attention_map(attn_data, target_shape)
            if attn_map is not None:
                start_col = idx * W
                end_col = start_col + W
                combined_image[:, start_col:end_col] = attn_map
        
        ax.imshow(combined_image, cmap='gray', aspect='auto')
        ax.set_title(f'{layer_name} ({len(maps_to_show)}/{n_maps} maps shown)', fontweight='bold')
        ax.axis('off')
        
        # Add separators
        for i in range(1, len(maps_to_show)):
            ax.axvline(x=i*W-0.5, color='white', linewidth=1)
    
    def _extract_multiple_attention_maps(self, attn_weights):
        """Extract multiple attention maps from various tensor formats."""
        try:
            if isinstance(attn_weights, list):
                # Handle list of tensors
                all_maps = []
                for item in attn_weights:
                    maps = self._extract_multiple_attention_maps(item)
                    all_maps.extend(maps)
                return all_maps
            
            if not isinstance(attn_weights, torch.Tensor):
                return []
                
            attn_weights = attn_weights.cpu()
            
            # Handle different tensor shapes
            if len(attn_weights.shape) == 4:  # [batch, heads, query, key]
                batch, heads, query, key = attn_weights.shape
                maps = []
                for b in range(batch):
                    for h in range(heads):
                        # Each attention head produces one map
                        attn_map = attn_weights[b, h, :, :].numpy()
                        maps.append(attn_map)
                return maps
                
            elif len(attn_weights.shape) == 3:  # [heads, query, key] or [batch, query, key]
                dim0, query, key = attn_weights.shape
                maps = []
                for i in range(dim0):
                    attn_map = attn_weights[i, :, :].numpy()
                    maps.append(attn_map)
                return maps
                
            elif len(attn_weights.shape) == 2:  # [query, key]
                return [attn_weights.numpy()]
                
            return []
            
        except Exception as e:
            print(f"❌ Error extracting attention maps: {e}")
            return []
    
    def _create_averaged_attention_map(self, attention_maps, target_shape):
        """Create averaged attention map from multiple maps."""
        if not attention_maps:
            return None
            
        resized_maps = []
        for attn_map in attention_maps:
            resized = self._resize_attention_map(attn_map, target_shape)
            if resized is not None:
                resized_maps.append(resized)
        
        if not resized_maps:
            return None
            
        # Average all maps
        avg_map = np.mean(resized_maps, axis=0)
        return avg_map
    
    def _resize_attention_map(self, attn_map, target_shape):
        """Resize attention map to target shape."""
        try:
            if attn_map.ndim != 2:
                return None
                
            H, W = target_shape
            current_H, current_W = attn_map.shape
            
            if (current_H, current_W) == (H, W):
                return attn_map
            
            # Normalize to [0, 1]
            attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            # Resize using PIL
            attn_img = Image.fromarray((attn_norm * 255).astype(np.uint8))
            resized_img = attn_img.resize((W, H), Image.BICUBIC)
            resized_map = np.array(resized_img) / 255.0
            
            return resized_map
            
        except Exception as e:
            print(f"❌ Error resizing attention map: {e}")
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


def find_attention_dirs(root_logs):
    """Find all attention_maps directories."""
    for dirpath, dirnames, _ in os.walk(root_logs):
        if os.path.basename(dirpath) == 'attention_maps':
            yield dirpath


def main():
    parser = argparse.ArgumentParser(description="Perceiver-style attention visualization")
    parser.add_argument('--logs_dir', type=str, default='logs', help='Root logs directory')
    parser.add_argument('--experiment', type=str, help='Specific experiment to visualize')
    parser.add_argument('--epoch', type=int, help='Specific epoch to visualize')
    parser.add_argument('--epochs', nargs='+', type=int, default=[1, 21, 41, 61, 81, 101, 120], 
                       help='Multiple epochs to visualize')
    parser.add_argument('--output_dir', type=str, default='perceiver_visualizations',
                       help='Output directory for visualizations')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find logs directory
    root_logs = Path(args.logs_dir)
    if not root_logs.exists():
        print(f"❌ Logs directory not found: {root_logs}")
        return
    
    print(f"📂 Using logs directory: {root_logs}")
    
    # Initialize visualizer
    visualizer = PerceiverAttentionVisualizer()
    
    # Find attention directories
    found_dirs = list(find_attention_dirs(root_logs))
    if not found_dirs:
        print(f"❌ No attention_maps directories found under {root_logs}")
        return
    
    print(f"📁 Found {len(found_dirs)} experiments with attention maps")
    
    # Filter by experiment if specified
    if args.experiment:
        found_dirs = [d for d in found_dirs if args.experiment in str(d)]
        if not found_dirs:
            print(f"❌ No experiments found matching: {args.experiment}")
            return
    
    # Process each experiment
    for attn_dir in found_dirs:
        exp_name = Path(attn_dir).parent.name
        print(f"\n🔍 Processing experiment: {exp_name}")
        
        # Find available epochs
        files = os.listdir(attn_dir)
        available_epochs = []
        for f in files:
            if f.startswith('epoch_') and f.endswith('_cross_attn_weights.pt'):
                epoch_num = int(f.split('_')[1])
                available_epochs.append(epoch_num)
        
        available_epochs = sorted(available_epochs)
        if not available_epochs:
            print(f"  ⚠️ No attention files found in {attn_dir}")
            continue
            
        print(f"  📊 Available epochs: {available_epochs}")
        
        # Determine epochs to process
        if args.epoch:
            epochs_to_process = [args.epoch] if args.epoch in available_epochs else []
        else:
            epochs_to_process = [e for e in args.epochs if e in available_epochs]
        
        if not epochs_to_process:
            print(f"  ⚠️ No matching epochs found")
            continue
            
        print(f"  🎯 Processing epochs: {epochs_to_process}")
        
        # Process each epoch
        for epoch in epochs_to_process:
            cross_file = f'epoch_{epoch}_cross_attn_weights.pt'
            orig_file = f'epoch_{epoch}_original_image_tensor.pt'
            cross_path = os.path.join(attn_dir, cross_file)
            orig_path = os.path.join(attn_dir, orig_file)
            
            if not (os.path.isfile(cross_path) and os.path.isfile(orig_path)):
                print(f"  ❌ Missing files for epoch {epoch}")
                continue
            
            try:
                # Load data
                print(f"    📥 Loading epoch {epoch} data...")
                attn_weights = torch.load(cross_path, map_location='cpu')
                img_tensor = torch.load(orig_path, map_location='cpu')
                
                # Create Perceiver-style visualization
                save_path = output_dir / f'{exp_name}_epoch_{epoch}_perceiver_style.png'
                fig = visualizer.create_perceiver_style_visualization(
                    attn_weights, img_tensor, 
                    title=f'{exp_name} - Epoch {epoch}', 
                    save_path=str(save_path)
                )
                
                if fig:
                    plt.close(fig)  # Free memory
                    print(f"    ✅ Created: {save_path.name}")
                else:
                    print(f"    ❌ Failed to create visualization for epoch {epoch}")
                    
            except Exception as e:
                print(f"    ❌ Error processing epoch {epoch}: {e}")
                continue
    
    print(f"\n🎉 Perceiver-style visualizations complete!")
    print(f"📁 Check results in: {output_dir}")
    print("\n💡 These visualizations show:")
    print("  • Original CIFAR-10 images")
    print("  • Attention overlays highlighting focus areas") 
    print("  • Grid of individual attention maps (checkerboard patterns)")
    print("  • Statistical analysis of attention patterns")


if __name__ == '__main__':
    main()
