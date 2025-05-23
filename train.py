# train.py
# Main training loop for the Perceiver model.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import os
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Project specific imports
from src.config.base_cfg import get_base_config
from src.data.cifar10 import CIFAR10PerceiverDataModule
from src.data.modelnet40 import ModelNet40PerceiverDataModule
from src.perceiver.perceiver import Perceiver
from src.utils.scheduler import get_scheduler
from src.utils.logger import BaseLogger
import torch_optimizer as custom_optim # For LAMB optimizer

def main(args):
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup Logger
    logger = BaseLogger(
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
        use_tensorboard=args.use_tensorboard,
        use_wandb=args.use_wandb,
        wandb_project_name=args.wandb_project_name
    )
    # Log hyperparameters
    logger.log_hparams(vars(args), {"initial_metric": 0})
    
    # Save configuration to a text file for easy reference
    experiment_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)  # Create experiment directory if it doesn't exist
    
    # Initialize data module
    input_dim, num_classes = None, 0
    
    # Setup data module based on dataset choice
    if args.dataset == 'cifar10':
        data_module = CIFAR10PerceiverDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size_cifar10,
            num_workers=args.num_workers,
            fourier_dim=args.cifar10_fourier_bands,
            max_frequencies=args.cifar10_max_freq,
            circular_pos_encoding=True,
            randaugment_num_ops=2,
            randaugment_magnitude=9
        )
        num_classes = 10
        args.batch_size = args.batch_size_cifar10
    elif args.dataset == 'modelnet40':
        data_module = ModelNet40PerceiverDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size_modelnet40,
            num_workers=args.num_workers,
            num_points=args.modelnet40_num_points,
            fourier_dim=args.modelnet40_fourier_bands,
            max_frequencies=args.modelnet40_max_freq,
            num_frequency_bands=6,
            augment_train=True
        )
        num_classes = 40
        args.batch_size = args.batch_size_modelnet40
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Setup data module
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # For CIFAR-10, get input dimension from the datamodule
    if args.dataset == 'cifar10':
        # Calculate patch size and other parameters
        patch_size = data_module.patch_size
        patch_dim = patch_size * patch_size * 3  # RGB patch
        input_dim = patch_dim + data_module.fourier_dim  # patch + positional encoding
    elif args.dataset == 'modelnet40':
        # For ModelNet40, input dimension is 3 (coordinates) + fourier_dim
        input_dim = 3 + data_module.fourier_dim

    # Calculate head_dim based on latent_dim and num_heads
    head_dim = args.latent_dim // args.num_heads
    
    # Temporary model instantiation just to calculate params (on CPU)
    temp_model = Perceiver(
        input_dim=input_dim,
        num_classes=num_classes,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        num_cross_attend_stages=args.num_cross_attend_stages,
        num_transformer_blocks=args.num_transformer_blocks,
        num_heads=args.num_heads, 
        head_dim=head_dim,
        mlp_ratio=4,
        dropout=args.dropout,
        output_pooling=args.output_pooling,
        save_attention_maps=args.save_attention_maps,
        weight_sharing=not args.no_weight_sharing
    )
    total_params = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
    del temp_model  # Free up memory
    
    config_file_path = os.path.join(experiment_dir, "config.txt")
    with open(config_file_path, "w") as f:
        f.write(f"PERCEIVER EXPERIMENT CONFIGURATION: {args.experiment_name}\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("COMMAND LINE ARGUMENTS:\n")
        f.write("-" * 20 + "\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
        
        f.write("\nMODEL ARCHITECTURE:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Model: Perceiver\n")
        f.write(f"Input dimension: {input_dim}\n")
        f.write(f"Number of latents: {args.num_latents}\n")
        f.write(f"Latent dimension: {args.latent_dim}\n")
        f.write(f"Number of cross-attention stages: {args.num_cross_attend_stages}\n")
        f.write(f"Number of transformer blocks: {args.num_transformer_blocks}\n")
        f.write(f"Number of attention heads: {args.num_heads}\n")
        f.write(f"Head dimension: {head_dim}\n")
        f.write(f"MLP ratio: 4\n")
        f.write(f"Weight sharing: {not args.no_weight_sharing}\n")
        f.write(f"Total trainable parameters: {total_params:,}\n")
        
        f.write("\nTRAINING CONFIGURATION:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"Learning rate: {args.lr}\n")
        f.write(f"Scheduler: {args.scheduler}\n")
        f.write(f"Number of epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        
        # Add model summary info
        f.write("\n" + "=" * 50 + "\n")
        f.write(f"Training start time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"Configuration saved to {config_file_path}")
    print(f"Dataset: {args.dataset}, Input dim: {input_dim}, Num classes: {num_classes}")
    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")

    # Set weight sharing flag (default is True, --no_weight_sharing disables it)
    weight_sharing = not args.no_weight_sharing
    
    # Initialize Model
    model = Perceiver(
        input_dim=input_dim,
        num_classes=num_classes,
        num_latents=args.num_latents,
        latent_dim=args.latent_dim,
        num_cross_attend_stages=args.num_cross_attend_stages,
        num_transformer_blocks=args.num_transformer_blocks,
        num_heads=args.num_heads, 
        head_dim=args.latent_dim // args.num_heads, # Calculate head_dim
        mlp_ratio=4, # Standard MLP ratio, can be made configurable
        dropout=args.dropout, # Use dropout from config
        output_pooling=args.output_pooling, # Use output pooling from config
        save_attention_maps=args.save_attention_maps, # Pass the flag
        weight_sharing=weight_sharing # Pass weight sharing flag
    ).to(device)
    
    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    # Optimizer
    if args.optimizer.lower() == 'lamb':
        optimizer = custom_optim.Lamb(model.parameters(), lr=args.lr)
        print(f"Using Lamb optimizer with lr={args.lr}")
    elif args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        print(f"Using Adam optimizer with lr={args.lr}")
    else: # Default or SGD
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        print(f"Using SGD optimizer with lr={args.lr}")

    # Scheduler
    scheduler = get_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        total_epochs=args.epochs,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        eta_min=args.eta_min_cosine
    )

    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    print(f"Mixed Precision Training (AMP): {'Enabled' if device.type == 'cuda' else 'Disabled (CPU fallback)'}")

    # Early stopping parameters
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    patience = 10  # Stop after 10 epochs without improvement
    
    # Tracking previous accuracies to calculate differences
    prev_train_acc = 0.0
    prev_val_acc = 0.0
    
    # Create checkpoints directory
    checkpoints_dir = os.path.join(args.log_dir, args.experiment_name, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoints_dir}")

    print("\nStarting Training...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch+1}/{args.epochs} ---")

        # Train one epoch
        avg_train_loss, avg_train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, args, scaler, data_module
        )
        # Convert accuracy to percentage and calculate difference
        train_acc_pct = avg_train_acc * 100
        train_diff = train_acc_pct - (prev_train_acc * 100)
        diff_sign = '+' if train_diff > 0 else ''
        print(f"Epoch {epoch+1} Train: Avg Loss: {avg_train_loss:.4f}, Avg Acc: {train_acc_pct:.2f}% ({diff_sign}{train_diff:.2f}%)")
        
        # Update previous accuracy for next epoch
        prev_train_acc = avg_train_acc
        
        logger.log_scalar("train/epoch_loss", avg_train_loss, epoch + 1)
        logger.log_scalar("train/epoch_accuracy", avg_train_acc, epoch + 1)

        # Save attention maps periodically if enabled
        if hasattr(args, 'save_attention_maps') and args.save_attention_maps and hasattr(args, 'attention_save_interval'):
            if epoch % args.attention_save_interval == 0 or epoch == args.epochs - 1:
                save_attention_maps(model, val_loader, device, epoch, args, data_module)
        
        # Validate one epoch
        if val_loader:
            avg_val_loss, avg_val_acc = validate_one_epoch(
                model, val_loader, criterion, device, epoch, logger, args, data_module
            )
            # Convert accuracy to percentage and calculate difference
            val_acc_pct = avg_val_acc * 100
            val_diff = val_acc_pct - (prev_val_acc * 100)
            diff_sign = '+' if val_diff > 0 else ''
            best_val_acc_pct = best_val_accuracy * 100  # Convert to percentage
            print(f"Epoch {epoch+1} Val: Avg Loss: {avg_val_loss:.4f}, Avg Acc: {val_acc_pct:.2f}% ({diff_sign}{val_diff:.2f}%), Top Acc: {best_val_acc_pct:.2f}%")
            
            # Update previous accuracy for next epoch
            prev_val_acc = avg_val_acc
            
            logger.log_scalar("val/epoch_loss", avg_val_loss, epoch + 1)
            logger.log_scalar("val/epoch_accuracy", avg_val_acc, epoch + 1)

            # Check if validation accuracy improved
            if avg_val_acc > best_val_accuracy:
                best_val_accuracy = avg_val_acc
                epochs_no_improve = 0  # Reset counter
                
                # Save the model (just the model's state_dict)
                best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved new best model to {best_model_path} with Val Acc: {avg_val_acc:.4f}")
                
                # Also save a full checkpoint (for resuming training if needed)
                checkpoint_path = os.path.join(checkpoints_dir, "last_checkpoint.pth")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': avg_val_acc,
                    'args': args
                }, checkpoint_path)
            else:
                # Increment epochs with no improvement
                epochs_no_improve += 1
                print(f"No improvement for {epochs_no_improve} epochs (best Val Acc: {best_val_accuracy:.4f})")
                
                # Check if early stopping criteria is met
                if epochs_no_improve >= patience:
                    print(f"\nEarly stopping! No improvement for {patience} consecutive epochs.")
                    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
                    break
        
        if scheduler:
            scheduler.step()
            logger.log_scalar("train/learning_rate", scheduler.get_last_lr()[0], epoch + 1)

    logger.close()
    print("\nTraining complete.")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")
    
    # Final evaluation with best model (optional)
    best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print("\nRunning final evaluation with best model...")
        # Load the best model
        model = load_best_model(model, device, best_model_path)
        
        # Run evaluation with best model
        with torch.no_grad():
            final_val_loss, final_val_acc = validate_one_epoch(
                model, val_loader, criterion, device, args.epochs, logger, args, data_module
            )
            print(f"Final evaluation - Best model: Loss: {final_val_loss:.4f}, Accuracy: {final_val_acc*100:.2f}%")
            logger.log_scalar("val/final_best_loss", final_val_loss, args.epochs)
            logger.log_scalar("val/final_best_accuracy", final_val_acc, args.epochs)

    # Save attention maps from the first validation sample if requested
    if args.save_attention_maps and val_loader and len(val_loader) > 0:
        print("\nSaving attention maps for one validation sample...")
        model.eval() # Ensure model is in eval mode
        
        # Get one batch from val_loader
        data_batch = next(iter(val_loader))
        if args.dataset == 'cifar10' or args.dataset == 'modelnet40':
            # Process batch through data module
            batch_dict = data_module.preprocess_batch(data_batch)
            data_sample = batch_dict['inputs']
            # For attention visualization, only use the first item
            single_data_item = data_sample[0:1].to(device)
        else:
            # Generic fallback
            data_sample, _ = data_batch
            single_data_item = data_sample[0:1].to(device)

        with torch.no_grad():
            # Use autocast for consistency
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                _ = model(single_data_item) # Run forward pass to populate model.attn_maps

        if hasattr(model, 'attn_maps') and model.attn_maps and len(model.attn_maps) > 0:
            attn_maps_save_dir = os.path.join(args.log_dir, args.experiment_name, "attention_maps")
            os.makedirs(attn_maps_save_dir, exist_ok=True)
            
            # Save the attention maps
            torch.save(model.attn_maps, os.path.join(attn_maps_save_dir, "cross_attention_maps_sample.pt"))
            
            # Also save the input sample that generated these maps
            torch.save(single_data_item.cpu(), os.path.join(attn_maps_save_dir, "input_sample_processed.pt"))
            
            # Save original image if dataset is CIFAR-10
            if args.dataset == 'cifar10' and 'original_images' in batch_dict:
                original_img_tensor = batch_dict['original_images'][0]  # Get first image
                torch.save(original_img_tensor, os.path.join(attn_maps_save_dir, "original_image_tensor.pt"))
                print("Saved original image tensor.")
            
            print(f"Saved attention maps and processed input sample to {attn_maps_save_dir}")
        else:
            print("No attention maps found in model. Ensure save_attention_maps=True in model config.")


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_num, logger, args, scaler, data_module):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch_num+1}/{args.epochs} [Train]")
    for batch_idx, batch in enumerate(progress_bar):
        # Process batch through data module
        if args.dataset == 'cifar10' or args.dataset == 'modelnet40':
            # Use the data module to preprocess the batch
            batch_dict = data_module.preprocess_batch(batch)
            data = batch_dict['inputs']
            target = batch_dict['labels']
        else:
            # Generic fallback for any batch format
            data, target = batch
        
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        
        # Use autocast for mixed precision training
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            output = model(data) 
            loss = criterion(output, target)
        
        # Use scaler for backwards pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * data.size(0)
        _, predicted = torch.max(output.data, 1)
        total_correct += (predicted == target).sum().item()
        total_samples += data.size(0)

        # Update progress bar
        current_loss = total_loss / total_samples
        current_acc = total_correct / total_samples
        progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")
        logger.log_scalar("train/batch_loss", loss.item(), epoch_num * len(train_loader) + batch_idx)
        logger.log_scalar("train/batch_accuracy", current_acc, epoch_num * len(train_loader) + batch_idx)
            
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc


def validate_one_epoch(model, val_loader, criterion, device, epoch_num, logger, args, data_module):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    # Collect all predictions and targets for advanced metrics
    all_predictions = []
    all_targets = []

    progress_bar = tqdm(val_loader, desc=f"Epoch {epoch_num+1}/{args.epochs} [Val]")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Process batch through data module
            if args.dataset == 'cifar10' or args.dataset == 'modelnet40':
                # Use the data module to preprocess the batch
                batch_dict = data_module.preprocess_batch(batch)
                data = batch_dict['inputs']
                target = batch_dict['labels']
            else:
                # Generic fallback for any batch format
                data, target = batch
            
            data, target = data.to(device), target.to(device)
            
            # Use autocast for mixed precision in validation too
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                output = model(data)
                loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
            
            # Save predictions and targets for metrics calculation
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            current_loss = total_loss / total_samples
            current_acc = total_correct / total_samples
            progress_bar.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")

    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    
    # Calculate and save advanced metrics if enabled
    if hasattr(args, 'save_metrics') and args.save_metrics:
        calculate_and_save_metrics(all_targets, all_predictions, epoch_num, args)
    
    return avg_loss, avg_acc


def load_best_model(model, device, model_path):
    """
    Load the best saved model weights from the checkpoint file.
    
    Args:
        model: The model to load weights into
        device: The device to load the model on
        model_path: Path to the saved model weights
        
    Returns:
        The model with loaded weights
    """
    if os.path.exists(model_path):
        print(f"Loading best model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        return model
    else:
        print(f"Warning: No saved model found at {model_path}")
        return model


def save_attention_maps(model, val_loader, device, epoch_num, args, data_module):
    """
    Save attention maps (cross-attention and self-attention) at specific epochs.
    
    Args:
        model: The Perceiver model
        val_loader: Validation data loader
        device: Device to run the model on
        epoch_num: Current epoch number
        args: Training arguments
    """
    print(f"\nSaving attention maps for epoch {epoch_num+1}...")
    model.eval()  # Set model to evaluation mode
    
    # Create attention maps directory
    attn_maps_save_dir = os.path.join(args.log_dir, args.experiment_name, "attention_maps")
    os.makedirs(attn_maps_save_dir, exist_ok=True)
    
    try:
        # Get first validation batch
        batch = next(iter(val_loader))
        if args.dataset == 'cifar10' or args.dataset == 'modelnet40':
            # Process batch through data module
            batch_dict = data_module.preprocess_batch(batch)
            data = batch_dict['inputs']
            target = batch_dict['labels']
        else:
            # Generic fallback
            data, target = batch
        
        single_data_item = data[0:1].to(device)  # Take just the first item
        
        with torch.no_grad():
            # Run forward pass with attention map saving enabled
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                _ = model(single_data_item)  # Forward pass to populate model.attn_maps
        
        # Save cross-attention maps
        if hasattr(model, 'attn_maps') and model.attn_maps and len(model.attn_maps) > 0:
            cross_attn_path = os.path.join(attn_maps_save_dir, f"epoch_{epoch_num+1}_cross_attn_weights.pt")
            torch.save(model.attn_maps, cross_attn_path)
            
            # Save self-attention maps if available
            if hasattr(model, 'self_attn_maps') and model.self_attn_maps and len(model.self_attn_maps) > 0:
                self_attn_path = os.path.join(attn_maps_save_dir, f"epoch_{epoch_num+1}_self_attn_weights.pt")
                torch.save(model.self_attn_maps, self_attn_path)
            
            # Save processed input sample
            processed_input_path = os.path.join(attn_maps_save_dir, f"epoch_{epoch_num+1}_processed_input.pt")
            torch.save(single_data_item.cpu(), processed_input_path)
            
            # For CIFAR-10, try to save original image
            if args.dataset == 'cifar10':
                try:
                    # Get a raw sample from the dataset
                    from torchvision.datasets import CIFAR10
                    from torchvision import transforms
                    
                    raw_dataset = CIFAR10(root=args.data_dir, train=False, download=False, transform=transforms.ToTensor())
                    original_img_tensor, _ = raw_dataset[0]  # Get the first test image
                    
                    original_image_path = os.path.join(attn_maps_save_dir, f"epoch_{epoch_num+1}_original_image_tensor.pt")
                    torch.save(original_img_tensor, original_image_path)
                except Exception as e:
                    print(f"Error saving original image: {e}")
                
            print(f"Attention maps for epoch {epoch_num+1} saved to: {attn_maps_save_dir}")
        else:
            print(f"No attention maps found in model for epoch {epoch_num+1}. Ensure save_attention_maps=True.")
    
    except Exception as e:
        print(f"Error saving attention maps: {e}")


def calculate_and_save_metrics(y_true, y_pred, epoch_num, args):
    """
    Calculate advanced metrics and save confusion matrix visualization.
    
    Args:
        y_true: Ground truth labels
        y_pred: Model predictions
        epoch_num: Current epoch number
        args: Training arguments
    """
    # Create metrics directory
    metrics_dir = os.path.join(args.log_dir, args.experiment_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Calculate metrics
    try:
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        f1 = f1_score(y_true, y_pred, average='macro')
        cm = confusion_matrix(y_true, y_pred)
        
        # Print metrics
        print("\n=== Advanced Metrics ===")
        print(f"Precision (macro): {precision:.4f}")
        print(f"Recall (macro): {recall:.4f}")
        print(f"F1 Score (macro): {f1:.4f}")
        
        # Save confusion matrix as heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - Epoch {epoch_num+1}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save the figure
        cm_filename = os.path.join(metrics_dir, f"confusion_matrix_epoch_{epoch_num+1}.png")
        plt.savefig(cm_filename, bbox_inches='tight')
        plt.close()
        
        print(f"Confusion matrix saved to: {cm_filename}")
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")


if __name__ == "__main__":
    parser = get_base_config()
    args = parser.parse_args()
    
    # Correct data_dir path to be relative to project root if it's './data'
    # Assuming this script is run from perceiver_project/
    if args.data_dir == './data':
        args.data_dir = os.path.abspath(args.data_dir)
        print(f"Resolved data_dir to absolute path: {args.data_dir}")

    main(args)
