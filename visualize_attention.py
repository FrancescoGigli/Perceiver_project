# visualize_attention.py

import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def overlay_attention_map(attn_weights, img_np, output_path, title):
    """
    Plot original | attention heatmap | overlay and save to disk.
    """
    # Handle different types of attention weights
    if isinstance(attn_weights, list):
        # Try to convert nested list structure to tensor
        try:
            attn_weights = torch.tensor(attn_weights)
        except ValueError:
            # If direct conversion fails, try to process the structure
            print(f"Complex attention structure detected, attempting to process...")
            # Recursively find the first tensor in the nested structure
            def find_first_tensor(data):
                if isinstance(data, torch.Tensor):
                    return data
                elif isinstance(data, list) and len(data) > 0:
                    for item in data:
                        result = find_first_tensor(item)
                        if result is not None:
                            return result
                return None
            
            tensor_found = find_first_tensor(attn_weights)
            if tensor_found is not None:
                attn_weights = tensor_found
                print(f"Found tensor with shape: {attn_weights.shape}")
            else:
                print(f"Warning: Could not find a tensor in the attention weights")
                return
    
    # Print some debug info
    print(f"Attention weights type: {type(attn_weights)}")
    print(f"Attention weights shape: {attn_weights.shape if hasattr(attn_weights, 'shape') else 'unknown'}")
    
    # Check if tensor has the expected dimensions
    if not hasattr(attn_weights, 'shape') or len(attn_weights.shape) < 3:
        print(f"Warning: Attention weights don't have the expected dimensions")
        if hasattr(attn_weights, 'shape') and len(attn_weights.shape) == 2:
            # If 2D, add a dimension to make it 3D for processing
            attn_weights = attn_weights.unsqueeze(0)
        else:
            return
    
    # Handle various dimension arrangements safely
    try:
        # First, ensure we're working with tensor on CPU
        attn_weights = attn_weights.cpu()
        
        # Try to average over dimensions that exist
        if len(attn_weights.shape) >= 3:
            # For shape [batch, heads, seq_len, seq_len] or similar
            attn_mean = attn_weights.mean(dim=1)
            if len(attn_mean.shape) >= 3:
                attn_mean = attn_mean.mean(dim=1)
            if len(attn_mean.shape) >= 2 and attn_mean.shape[0] == 1:
                attn_mean = attn_mean.squeeze(0)
        else:
            # Fallback for unusual shapes
            attn_mean = attn_weights
        
        attn_mean = attn_mean.numpy()
    except Exception as e:
        print(f"Error processing attention weights: {e}")
        return
    H, W, _ = img_np.shape
    src_len = H * W

    # Print debug info about sizes
    print(f"Image shape: {img_np.shape}, Attention mean shape/size: {attn_mean.shape if hasattr(attn_mean, 'shape') else 'scalar'}, Size: {attn_mean.size}")
    
    # Handle different attention map shapes/sizes
    try:
        # Check if 1D or flattened attention
        if isinstance(attn_mean, np.ndarray) and attn_mean.ndim == 1 and attn_mean.size == src_len:
            # Perfect match, just reshape
            attn_map = attn_mean.reshape(H, W)
        
        # Check if it's already 2D with correct dimensions
        elif isinstance(attn_mean, np.ndarray) and attn_mean.ndim == 2 and attn_mean.shape == (H, W):
            # Already in the right shape
            attn_map = attn_mean
            
        # Otherwise, resize the attention map to match image dimensions
        else:
            if isinstance(attn_mean, np.ndarray) and attn_mean.ndim >= 2:
                # If multi-dimensional, flatten to 2D first by taking the first slice if needed
                if attn_mean.ndim > 2:
                    print(f"Reducing dimensions from {attn_mean.ndim}D to 2D")
                    if attn_mean.shape[0] == 1:  # If batch dimension is 1
                        attn_mean = attn_mean[0]
                    else:
                        # Take the mean across the first dimension
                        attn_mean = attn_mean.mean(axis=0)
                
                # Convert to image and resize
                norm_attn = (attn_mean - attn_mean.min()) / (attn_mean.max() - attn_mean.min() + 1e-8)
                heatmap = Image.fromarray((norm_attn * 255).astype(np.uint8))
                heatmap = heatmap.resize((W, H), Image.BICUBIC)
                attn_map = np.array(heatmap) / 255.0
            
            elif isinstance(attn_mean, np.ndarray) and attn_mean.ndim == 1:
                # For 1D array with wrong size, reshape to square and then resize
                side = int(np.sqrt(attn_mean.size))
                reshaped = attn_mean[:side*side].reshape(side, side)
                norm_attn = (reshaped - reshaped.min()) / (reshaped.max() - reshaped.min() + 1e-8)
                heatmap = Image.fromarray((norm_attn * 255).astype(np.uint8))
                heatmap = heatmap.resize((W, H), Image.BICUBIC)
                attn_map = np.array(heatmap) / 255.0
            
            else:
                # Last resort: create a dummy attention map
                print("Warning: Could not process attention map properly, using uniform map")
                attn_map = np.ones((H, W)) * 0.5
    
    except Exception as e:
        print(f"Error processing attention map for visualization: {e}")
        # Create a dummy attention map
        attn_map = np.ones((H, W)) * 0.5

    # build overlay
    cmap = plt.get_cmap('jet')
    colored = cmap(attn_map)[:,:,:3]
    overlay = (1-0.5)*img_np + 0.5*colored
    overlay = np.clip(overlay,0,1)

    # plot side-by-side
    fig, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].imshow(img_np);           axs[0].axis('off'); axs[0].set_title('Original')
    axs[1].imshow(attn_map,cmap='jet'); axs[1].axis('off'); axs[1].set_title('Attention')
    axs[2].imshow(overlay);           axs[2].axis('off'); axs[2].set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"✔ Saved: {output_path}")

def find_attention_dirs(root_logs):
    """
    Walk root_logs and yield every path named 'attention_maps'.
    """
    for dirpath, dirnames, _ in os.walk(root_logs):
        if os.path.basename(dirpath)=='attention_maps':
            yield dirpath

def main():
    # 1) autodetect logs dir
    candidates = ['logs', os.path.join('perceiver_project','logs')]
    for c in candidates:
        if os.path.isdir(c):
            root_logs = c
            break
    else:
        print("❌ Cannot find a 'logs/' directory. Checked:", candidates)
        return

    print(f"Using logs directory: {root_logs}")

    # 2) scan all attention_maps/
    any_dir=False
    for attn_dir in find_attention_dirs(root_logs):
        any_dir=True
        print("\n▶ Found attention_maps/ at:", attn_dir)
        files = os.listdir(attn_dir)
        # detect epochs via cross_attn
        pat = re.compile(r'epoch_(\d+)_cross_attn_weights\.pt')
        epochs = sorted(int(m.group(1)) for f in files if (m:=pat.match(f)))
        if not epochs:
            print("  ⚠️ No 'epoch_<N>_cross_attn_weights.pt' here, skipping.")
            continue
        print("  Epochs:", epochs)

        for e in epochs:
            print(f"  • Epoch {e}")
            cross_f = f'epoch_{e}_cross_attn_weights.pt'
            orig_f  = f'epoch_{e}_original_image_tensor.pt'
            cross_p = os.path.join(attn_dir,cross_f)
            orig_p  = os.path.join(attn_dir,orig_f)

            if not os.path.isfile(cross_p):
                print(f"    ❌ Missing {cross_f}, skip")
                continue
            if not os.path.isfile(orig_p):
                print(f"    ❌ Missing {orig_f}, skip")
                continue

            try:
                # Load attention weights - don't use weights_only as it might change the structure
                try:
                    attn = torch.load(cross_p)  # [1, heads, tgt_len, src_len]
                    print(f"    Loaded attention weights: {type(attn)}")
                except Exception as e:
                    print(f"    ⚠️ Error loading attention weights: {e}")
                    continue
                
                # Load image tensor
                try:
                    img_t = torch.load(orig_p)  # [3, H, W]
                except Exception as e:
                    print(f"    ⚠️ Error loading image tensor: {e}")
                    continue
                
                # Print debug info
                print(f"    Image tensor type: {type(img_t)}")
                if isinstance(img_t, torch.Tensor):
                    print(f"    Image tensor shape: {img_t.shape}")
                elif isinstance(img_t, list):
                    print(f"    Image tensor is a list of length {len(img_t)}")
                    if len(img_t) > 0:
                        print(f"    First element type: {type(img_t[0])}")
                
                # Ensure img_t is a tensor with the right format
                try:
                    if not isinstance(img_t, torch.Tensor):
                        # Handle list structure
                        if isinstance(img_t, list) and len(img_t) > 0:
                            # Recursive function to find the first tensor or convertible array
                            def find_first_convertible(data, depth=0, max_depth=5):
                                if depth > max_depth:  # Prevent infinite recursion
                                    return None
                                
                                if isinstance(data, torch.Tensor):
                                    return data
                                elif isinstance(data, (list, tuple)) and len(data) > 0:
                                    for item in data:
                                        result = find_first_convertible(item, depth+1, max_depth)
                                        if result is not None:
                                            return result
                                elif isinstance(data, (np.ndarray, list)) and len(data) > 0:
                                    try:
                                        return torch.tensor(data)
                                    except:
                                        pass
                                return None
                            
                            tensor_found = find_first_convertible(img_t)
                            if tensor_found is not None:
                                img_t = tensor_found
                                print(f"    Found convertible tensor with shape: {img_t.shape}")
                            else:
                                try:
                                    # Fallback: try converting the first element
                                    img_t = torch.tensor(img_t[0] if isinstance(img_t[0], (list, tuple)) and len(img_t[0]) > 0 else img_t)
                                    print(f"    Converted to tensor with shape: {img_t.shape}")
                                except:
                                    print(f"    ⚠️ Could not convert to tensor, skipping")
                                    continue
                        else:
                            print(f"    ⚠️ Unexpected image tensor format, skipping")
                            continue
                    
                    # Ensure tensor is on CPU
                    img_t = img_t.cpu()
                    
                    # Handle different tensor dimensions
                    if len(img_t.shape) == 4 and img_t.shape[0] == 1:  # [1, C, H, W]
                        img_t = img_t.squeeze(0)
                    
                    if len(img_t.shape) != 3:
                        print(f"    ⚠️ Unexpected tensor shape: {img_t.shape}, expected [C, H, W]")
                        if len(img_t.shape) == 2:  # [H, W]
                            # Convert grayscale to RGB
                            img_t = img_t.unsqueeze(0).repeat(3, 1, 1)
                        else:
                            print(f"    Cannot convert tensor to image format, skipping")
                            continue
                    
                    # Now convert to numpy array with channels last format
                    img_np = img_t.permute(1,2,0).cpu().numpy()
                    
                    # Normalize if needed
                    if img_np.max() > 1.0 or img_np.min() < 0.0:
                        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                
                except Exception as ex:
                    print(f"    ⚠️ Error processing image tensor: {ex}")
                    continue
                
                out_png = os.path.join(attn_dir, f'epoch_{e}_cross_overlay.png')
                overlay_attention_map(attn, img_np, out_png, title=f'E{e} Cross-Attn')
            
            except Exception as ex:
                print(f"    ⚠️ Error processing epoch {e}: {ex}")
                import traceback
                print(f"    Traceback: {traceback.format_exc()}")
                continue

    if not any_dir:
        print("❌ No attention_maps/ directories found under", root_logs)
    else:
        print("\n✅ Done.")

if __name__=='__main__':
    main()
