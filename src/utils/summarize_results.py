import os
import torch
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger()

def summarize_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return None
    try:
        # Load just the essential metadata to avoid loading heavy weights
        # Actually torch.load loads everything unless we're clever, but Perceiver isn't huge.
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        
        epoch = ckpt.get('epoch', 'N/A')
        val_acc = ckpt.get('val_accuracy', 'N/A')
        # Some older checkpoints might store loss differently or not at all
        val_loss = ckpt.get('val_loss', 'N/A')
        
        return {
            'Epoch': epoch,
            'Val Acc': f"{val_acc:.4f}" if isinstance(val_acc, float) else val_acc,
            'Val Loss': f"{val_loss:.4f}" if isinstance(val_loss, float) else val_loss
        }
    except Exception as e:
        return {'Error': str(e)}

def main():
    experiments = [
        ("MLM Pre-training (WikiText-103)", "logs/mlm_wikitext103_optimized_batch32/checkpoints/best_model.pt"), # Or last_checkpoint
        ("SST-2 Fine-tune (v4 - First Run)", "logs/glue_sst2_finetune_v4/checkpoints/last_checkpoint.pth"),
        ("SST-2 Fine-tune (v5 - Latest)", "logs/glue_sst2_finetune_v5/checkpoints/last_checkpoint.pth"),
    ]

    print(f"{'Experiment':<40} | {'Epoch':<10} | {'Val Acc':<10} | {'Status':<15}")
    print("-" * 85)

    for name, path in experiments:
        data = summarize_checkpoint(path)
        if data:
            if 'Error' in data:
                print(f"{name:<40} | {'ERROR':<10} | {'N/A':<10} | {data['Error']:<15}")
            else:
                epoch_str = str(data['Epoch'])
                acc_str = str(data['Val Acc'])
                print(f"{name:<40} | {epoch_str:<10} | {acc_str:<10} | {'Completed'}")
        else:
             print(f"{name:<40} | {'N/A':<10} | {'N/A':<10} | {'Not Found'}")

if __name__ == "__main__":
    main()
