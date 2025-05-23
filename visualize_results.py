# visualize_results.py

import os
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
import argparse

def plot_accuracy_graph(metrics, output_dir, title_suffix=''):
    """
    Plot training and test accuracy per epoch.

    Args:
        metrics (list of dict): Each dict has keys ['epoch', 'train_accuracy', 'test_accuracy'].
        output_dir (str): Directory where to save the plot.
        title_suffix (str): Optional suffix for the plot title and filename.
    """
    epochs = [m['epoch'] for m in metrics]
    train_acc = [m['train_accuracy'] for m in metrics]
    test_acc  = [m['test_accuracy'] for m in metrics]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_acc, label='Train Accuracy')
    plt.plot(epochs, test_acc,  label='Test Accuracy')
    plt.title(f'Accuracy per Epoch{title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'accuracy_plot{title_suffix}.png')
    plt.savefig(out_path)
    plt.close()
    print(f"[VIS] Saved accuracy plot to {out_path}")


def visualize_attention_heatmaps(output_dir, selected_epochs):
    """
    For each selected epoch, load cross- and self-attention weights,
    average over heads and (for self-attn) blocks, and plot heatmaps.

    Args:
        output_dir (str): Root experiment directory containing 'attention_maps/'.
        selected_epochs (list of int): Epoch numbers to visualize.
    """
    attn_dir = os.path.join(output_dir, 'attention_maps')
    if not os.path.isdir(attn_dir):
        print(f"[WARN] Attention maps directory not found: {attn_dir}")
        return

    for epoch in selected_epochs:
        cross_path = os.path.join(attn_dir, f'epoch_{epoch}_cross_attn_weights.pt')
        self_path  = os.path.join(attn_dir, f'epoch_{epoch}_self_attn_weights.pt')

        if not os.path.isfile(cross_path):
            print(f"[WARN] Missing cross-attn for epoch {epoch}, skipping.")
            continue
        if not os.path.isfile(self_path):
            print(f"[WARN] Missing self-attn for epoch {epoch}, skipping self-attn heatmap.")

        # Load attention tensors
        cross = torch.load(cross_path)  # shape [batch, heads, tgt_len, src_len]
        self_list = torch.load(self_path)  # list of [batch, heads, tgt_len, tgt_len]

        # Convert to numpy and average
        cross_np = cross.detach().cpu().numpy()
        # average over batch and heads: result shape [tgt_len, src_len]
        cross_avg = np.mean(cross_np, axis=(0,1))

        # For self-attn, average over batch, heads, and blocks
        self_np_list = [w.detach().cpu().numpy() for w in self_list]
        self_avg_blocks = [np.mean(w, axis=(0,1)) for w in self_np_list]  # each [tgt_len, tgt_len]
        self_avg = np.mean(self_avg_blocks, axis=0)

        # Plot side-by-side heatmaps
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cross_avg, cmap='viridis', ax=axs[0])
        axs[0].set_title(f'Cross-Attn Epoch {epoch}')
        axs[0].set_xlabel('Input Tokens')
        axs[0].set_ylabel('Latent Tokens')

        sns.heatmap(self_avg, cmap='viridis', ax=axs[1])
        axs[1].set_title(f'Self-Attn Epoch {epoch}')
        axs[1].set_xlabel('Latent Tokens')
        axs[1].set_ylabel('Latent Tokens')

        plt.tight_layout()
        out_png = os.path.join(output_dir, f'attention_heatmaps_epoch_{epoch}.png')
        plt.savefig(out_png)
        plt.close()
        print(f"[VIS] Saved attention heatmaps for epoch {epoch} to {out_png}")


def overlay_attention_on_image(img_pil, attn_map, output_path, alpha=0.5):
    """
    Overlay a single-channel attention map on a PIL image.

    Args:
        img_pil (PIL.Image): The original RGB image.
        attn_map (ndarray): 2D array matching image size, containing values in [0,1].
        output_path (str): Path to save the overlay PNG.
        alpha (float): Blending factor for overlay.
    """
    # Normalize attention map to [0,1]
    attn_norm = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_uint8  = np.uint8(attn_norm * 255)
    # Resize to image dimensions
    attn_resized = np.array(Image.fromarray(attn_uint8).resize(img_pil.size, resample=Image.BILINEAR))
    # Apply a colormap
    cmap = plt.get_cmap('jet')
    colored = cmap(attn_resized / 255.0)[:, :, :3]
    colored_img = np.uint8(colored * 255)
    overlay = Image.blend(img_pil.convert("RGBA"), Image.fromarray(colored_img).convert("RGBA"), alpha)
    overlay.save(output_path)
    print(f"[VIS] Saved attention overlay to {output_path}")


def visualize_image_overlay(output_dir, selected_epoch):
    """
    Load a single example image and its cross-attention map for a given epoch,
    then overlay and save the result.

    Args:
        output_dir (str): Root experiment directory containing 'attention_maps/'.
        selected_epoch (int): The epoch number to visualize.
    """
    attn_dir = os.path.join(output_dir, 'attention_maps')
    # Load original unnormalized image tensor saved by train.py
    img_tensor_path = os.path.join(attn_dir, 'original_image_tensor.pt')
    if not os.path.isfile(img_tensor_path):
        print(f"[WARN] original_image_tensor.pt not found in {attn_dir}")
        return

    img_tensor = torch.load(img_tensor_path)  # shape [3,32,32], values [0,1]
    img_np = (img_tensor.permute(1,2,0).cpu().numpy())  # H×W×C
    img_pil = Image.fromarray(np.uint8(img_np * 255))

    # Load cross-attention for this epoch
    cross_path = os.path.join(attn_dir, f'epoch_{selected_epoch}_specific_cross_attn_weights.pt')
    if not os.path.isfile(cross_path):
        print(f"[WARN] Cross-attn map for epoch {selected_epoch} not found.")
        return
    cross = torch.load(cross_path)  # [1,heads,tgt_len,src_len]
    # Average to get [src_len]
    attn_vec = cross.mean(dim=1).mean(dim=1).squeeze(0).cpu().numpy()
    # reshape to 32×32
    attn_map = attn_vec.reshape(32,32)

    out_png = os.path.join(output_dir, f'overlay_epoch_{selected_epoch}.png')
    overlay_attention_on_image(img_pil, attn_map, out_png, alpha=0.5)


def main():
    parser = argparse.ArgumentParser(description="Plot training results and attention")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Experiment output directory (contains metrics.json and attention_maps/)")
    parser.add_argument('--epochs', nargs='+', type=int, default=[1,10,20,30,40,50,60,70,80,90,100],
                        help="Epochs to visualize heatmaps")
    parser.add_argument('--overlay_epoch', type=int, default=None,
                        help="Single epoch to create image overlay (requires original_image_tensor.pt)")
    args = parser.parse_args()

    # Load metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    if not os.path.isfile(metrics_path):
        print(f"[ERROR] metrics.json not found in {args.output_dir}")
        return
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # 1) Plot accuracy curves
    plot_accuracy_graph(metrics, args.output_dir)

    # 2) Plot attention heatmaps
    visualize_attention_heatmaps(args.output_dir, args.epochs)

    # 3) Optional: overlay attention on a sample image
    if args.overlay_epoch is not None:
        visualize_image_overlay(args.output_dir, args.overlay_epoch)


if __name__ == '__main__':
    main()
