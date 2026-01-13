#!/usr/bin/env python3
"""
ModelNet40 Augmentation Study Script
==================================

This script runs 3 experiments to study the effect of different data augmentations 
on ModelNet40 classification performance using the Perceiver model.

Target: Reproduce paper results (85.7% top-1 accuracy) and test augmentation impact.

Experiments:
1. Baseline (scale only): Paper configuration - target 85.7%
2. + Translation: Add random translation (-0.02 to +0.02)  
3. + Rotation: Add random rotation around random axis

All experiments use identical architecture and training settings from the paper.
"""

import subprocess
import time
import os
import json
import datetime
from pathlib import Path
import torch

class ModelNet40AugmentationStudy:
    def __init__(self, log_dir="logs", dry_run=False):
        """
        Initialize ModelNet40 augmentation study runner.
        
        Args:
            log_dir: Directory for saving logs
            dry_run: If True, only print commands without executing
        """
        self.log_dir = Path(log_dir)
        self.dry_run = dry_run
        self.results = {}
        self.start_time = datetime.datetime.now()
        
        # Create logs directory
        self.log_dir.mkdir(exist_ok=True)
        
        # Paper specifications for ModelNet40
        self.paper_config = {
            "dataset": "modelnet40",
            "num_points": 2048,  # ~2000 points as specified
            "batch_size": 512,
            "optimizer": "lamb", 
            "lr": 0.001,  # 1e-3 constant
            "scheduler": "none",  # Constant LR
            "epochs": 150,  # Allow for ~50k steps
            "fourier_bands": 64,
            "fourier_max_freq": 1120.0,  # 10x ImageNet frequency
            "num_latents": 128,
            "latent_dim": 512,
            "num_cross_attend_stages": 2,  # Paper specification
            "num_transformer_blocks": 6,  # Paper specification  
            "num_heads": 8,
            "dropout": 0.1,
            "target_accuracy": 85.7  # Paper result
        }
        
        # Define the 3 experiments
        self.experiments = {
            "modelnet40_baseline": {
                "description": "Baseline (scale only) - Paper configuration",
                "priority": 1,
                "augmentation": "scale_only",
                "expected_result": "Target: 85.7% (paper result)",
                "cmd_args": [
                    "--experiment_name", "modelnet40_baseline",
                    "--dataset", "modelnet40",
                    "--modelnet40_num_points", str(self.paper_config["num_points"]),
                    "--modelnet40_fourier_bands", str(self.paper_config["fourier_bands"]),
                    "--modelnet40_max_freq", str(self.paper_config["fourier_max_freq"]),
                    "--batch_size_modelnet40", str(self.paper_config["batch_size"]),
                    "--num_latents", str(self.paper_config["num_latents"]),
                    "--latent_dim", str(self.paper_config["latent_dim"]),
                    "--num_cross_attend_stages", str(self.paper_config["num_cross_attend_stages"]),
                    "--num_transformer_blocks", str(self.paper_config["num_transformer_blocks"]),
                    "--num_heads", str(self.paper_config["num_heads"]),
                    "--dropout", str(self.paper_config["dropout"]),
                    "--optimizer", self.paper_config["optimizer"],
                    "--lr", str(self.paper_config["lr"]),
                    "--scheduler", self.paper_config["scheduler"],
                    "--epochs", str(self.paper_config["epochs"]),
                    "--save_attention_maps",
                    "--save_metrics",
                    "--use_tensorboard"
                ]
            },
            
            "modelnet40_with_translation": {
                "description": "Scale + Translation (-0.02 to +0.02)",
                "priority": 2,
                "augmentation": "scale_translation",
                "expected_result": "Expected: ≤85.7% (paper states translation doesn't improve)",
                "cmd_args": [
                    "--experiment_name", "modelnet40_with_translation", 
                    "--dataset", "modelnet40",
                    "--modelnet40_num_points", str(self.paper_config["num_points"]),
                    "--modelnet40_fourier_bands", str(self.paper_config["fourier_bands"]),
                    "--modelnet40_max_freq", str(self.paper_config["fourier_max_freq"]),
                    "--batch_size_modelnet40", str(self.paper_config["batch_size"]),
                    "--num_latents", str(self.paper_config["num_latents"]),
                    "--latent_dim", str(self.paper_config["latent_dim"]),
                    "--num_cross_attend_stages", str(self.paper_config["num_cross_attend_stages"]),
                    "--num_transformer_blocks", str(self.paper_config["num_transformer_blocks"]),
                    "--num_heads", str(self.paper_config["num_heads"]),
                    "--dropout", str(self.paper_config["dropout"]),
                    "--optimizer", self.paper_config["optimizer"],
                    "--lr", str(self.paper_config["lr"]),
                    "--scheduler", self.paper_config["scheduler"],
                    "--epochs", str(self.paper_config["epochs"]),
                    "--save_attention_maps",
                    "--save_metrics",
                    "--use_tensorboard"
                ]
            },
            
            "modelnet40_with_rotation": {
                "description": "Scale + Random Rotation", 
                "priority": 3,
                "augmentation": "scale_rotation",
                "expected_result": "Expected: ≤85.7% (paper states rotation doesn't improve)",
                "cmd_args": [
                    "--experiment_name", "modelnet40_with_rotation",
                    "--dataset", "modelnet40", 
                    "--modelnet40_num_points", str(self.paper_config["num_points"]),
                    "--modelnet40_fourier_bands", str(self.paper_config["fourier_bands"]),
                    "--modelnet40_max_freq", str(self.paper_config["fourier_max_freq"]),
                    "--batch_size_modelnet40", str(self.paper_config["batch_size"]),
                    "--num_latents", str(self.paper_config["num_latents"]),
                    "--latent_dim", str(self.paper_config["latent_dim"]),
                    "--num_cross_attend_stages", str(self.paper_config["num_cross_attend_stages"]),
                    "--num_transformer_blocks", str(self.paper_config["num_transformer_blocks"]),
                    "--num_heads", str(self.paper_config["num_heads"]),
                    "--dropout", str(self.paper_config["dropout"]),
                    "--optimizer", self.paper_config["optimizer"],
                    "--lr", str(self.paper_config["lr"]),
                    "--scheduler", self.paper_config["scheduler"],
                    "--epochs", str(self.paper_config["epochs"]),
                    "--save_attention_maps", 
                    "--save_metrics",
                    "--use_tensorboard"
                ]
            }
        }
    
    def print_header(self):
        """Print script header and info.""" 
        print("=" * 80)
        print("🧠 MODELNET40 AUGMENTATION STUDY - Perceiver")
        print("=" * 80)
        print(f"📅 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Log Directory: {self.log_dir}")
        print(f"🧪 Total Experiments: {len(self.experiments)}")
        print(f"🎯 Target Accuracy: {self.paper_config['target_accuracy']}%")
        print(f"🖥️  Mode: {'DRY RUN' if self.dry_run else 'EXECUTION'}")
        print("=" * 80)
        print()
        
        # Print paper specifications
        print("📋 PAPER SPECIFICATIONS:")
        print("-" * 50)
        print(f"Dataset: ModelNet40 (9,843 train / 2,468 test)")
        print(f"Input: ~{self.paper_config['num_points']} points (x,y,z) per object")
        print(f"Architecture: {self.paper_config['num_cross_attend_stages']} cross-att + {self.paper_config['num_transformer_blocks']} self-att")
        print(f"PE: Fourier features, freq_max={self.paper_config['fourier_max_freq']}, {self.paper_config['fourier_bands']} bands")
        print(f"Training: batch_size={self.paper_config['batch_size']}, LAMB, LR={self.paper_config['lr']} (constant)")
        print(f"Target Performance: {self.paper_config['target_accuracy']}% top-1 accuracy")
        print()
    
    def print_experiment_plan(self):
        """Print the execution plan."""
        print("📋 EXPERIMENT PLAN:")
        print("-" * 50)
        
        # Sort by priority
        sorted_exps = sorted(self.experiments.items(), key=lambda x: x[1]['priority'])
        
        for i, (exp_name, config) in enumerate(sorted_exps, 1):
            print(f"{i}. {exp_name}")
            print(f"   📝 {config['description']}")
            print(f"   🔧 Augmentation: {config['augmentation']}")
            print(f"   🎯 {config['expected_result']}")
            print()
        print("-" * 50)
        print()
    
    def create_custom_train_script(self, exp_name, config):
        """Create a custom training script for each experiment with specific data module configuration."""
        
        # Map experiment to data module configuration
        augmentation_configs = {
            "modelnet40_baseline": {
                "use_translation": False,
                "use_rotation": False
            },
            "modelnet40_with_translation": {
                "use_translation": True,
                "use_rotation": False
            },
            "modelnet40_with_rotation": {
                "use_translation": False, 
                "use_rotation": True
            }
        }
        
        aug_config = augmentation_configs.get(exp_name, {"use_translation": False, "use_rotation": False})
        
        # Create custom training script content
        script_content = f'''#!/usr/bin/env python3
"""
Custom training script for {exp_name}
Generated automatically by ModelNet40 augmentation study.
"""

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
from src.data.modelnet40 import ModelNet40PerceiverDataModule
from src.perceiver.perceiver import Perceiver
from src.utils.scheduler import get_scheduler
from src.utils.logger import BaseLogger
import torch_optimizer as custom_optim

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {{device}}")
    
    # Experiment configuration
    exp_name = "{exp_name}"
    log_dir = "logs"
    
    # Paper-based configuration for ModelNet40
    config = {{
        'dataset': 'modelnet40',
        'data_dir': './data',
        'batch_size': {self.paper_config["batch_size"]},
        'num_workers': 4,
        'num_points': {self.paper_config["num_points"]},
        'fourier_bands': {self.paper_config["fourier_bands"]}, 
        'fourier_max_freq': {self.paper_config["fourier_max_freq"]},
        'num_latents': {self.paper_config["num_latents"]},
        'latent_dim': {self.paper_config["latent_dim"]},
        'num_cross_attend_stages': {self.paper_config["num_cross_attend_stages"]},
        'num_transformer_blocks': {self.paper_config["num_transformer_blocks"]},
        'num_heads': {self.paper_config["num_heads"]},
        'dropout': {self.paper_config["dropout"]},
        'optimizer': '{self.paper_config["optimizer"]}',
        'lr': {self.paper_config["lr"]},
        'epochs': {self.paper_config["epochs"]},
        'use_translation': {aug_config["use_translation"]},
        'use_rotation': {aug_config["use_rotation"]},
        'target_accuracy': {self.paper_config["target_accuracy"]}
    }}
    
    # Setup Logger
    logger = BaseLogger(
        log_dir=log_dir,
        experiment_name=exp_name,
        use_tensorboard=True,
        use_wandb=False
    )
    logger.log_hparams(config, {{"initial_metric": 0}})
    
    # Save configuration
    experiment_dir = os.path.join(log_dir, exp_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Initialize ModelNet40 data module with specific augmentation
    data_module = ModelNet40PerceiverDataModule(
        data_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        num_points=config['num_points'],
        fourier_dim=config['fourier_bands'],
        max_frequencies=config['fourier_max_freq'],
        num_frequency_bands=6,
        augment_train=True,
        use_translation=config['use_translation'],
        translate_range=0.02,  # Paper specification
        use_rotation=config['use_rotation']
    )
    
    # Setup data
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    # Model configuration
    input_dim = 3 + config['fourier_bands']  # coordinates + fourier features
    num_classes = 40
    head_dim = config['latent_dim'] // config['num_heads']
    
    print(f"Input dimension: {{input_dim}}")
    print(f"Number of classes: {{num_classes}}")
    print(f"Train batches: {{len(train_loader)}}, Val batches: {{len(val_loader)}}")
    print(f"Augmentation - Translation: {{config['use_translation']}}, Rotation: {{config['use_rotation']}}")
    
    # Initialize Model
    model = Perceiver(
        input_dim=input_dim,
        num_classes=num_classes,
        num_latents=config['num_latents'],
        latent_dim=config['latent_dim'],
        num_cross_attend_stages=config['num_cross_attend_stages'],
        num_transformer_blocks=config['num_transformer_blocks'],
        num_heads=config['num_heads'],
        head_dim=head_dim,
        mlp_ratio=4,
        dropout=config['dropout'],
        output_pooling='mean',
        save_attention_maps=True,
        weight_sharing=True
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized with {{total_params:,}} trainable parameters.")
    
    # Optimizer
    if config['optimizer'].lower() == 'lamb':
        optimizer = custom_optim.Lamb(model.parameters(), lr=config['lr'])
    elif config['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    
    # No scheduler (constant LR as per paper)
    scheduler = None
    
    # Loss Function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize GradScaler for AMP
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    # Training tracking
    best_val_accuracy = 0.0
    epochs_no_improve = 0
    patience = 15  # Increased patience for ModelNet40
    
    # Create checkpoints directory
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    print(f"\\nStarting Training for {{exp_name}}...")
    print(f"Target Accuracy: {{config['target_accuracy']}}%")
    print("=" * 60)
    
    for epoch in range(config['epochs']):
        print(f"\\n--- Epoch {{epoch+1}}/{{config['epochs']}} ---")
        
        # Train one epoch
        avg_train_loss, avg_train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger, data_module, scaler
        )
        
        train_acc_pct = avg_train_acc * 100
        print(f"Epoch {{epoch+1}} Train: Loss: {{avg_train_loss:.4f}}, Acc: {{train_acc_pct:.2f}}%")
        
        logger.log_scalar("train/epoch_loss", avg_train_loss, epoch + 1)
        logger.log_scalar("train/epoch_accuracy", avg_train_acc, epoch + 1)
        
        # Validate one epoch
        avg_val_loss, avg_val_acc = validate_one_epoch(
            model, val_loader, criterion, device, epoch, logger, data_module
        )
        
        val_acc_pct = avg_val_acc * 100
        print(f"Epoch {{epoch+1}} Val: Loss: {{avg_val_loss:.4f}}, Acc: {{val_acc_pct:.2f}}%")
        print(f"Best Val Acc: {{best_val_accuracy*100:.2f}}%, Target: {{config['target_accuracy']}}%")
        
        logger.log_scalar("val/epoch_loss", avg_val_loss, epoch + 1)
        logger.log_scalar("val/epoch_accuracy", avg_val_acc, epoch + 1)
        
        # Check if validation accuracy improved
        if avg_val_acc > best_val_accuracy:
            best_val_accuracy = avg_val_acc
            epochs_no_improve = 0
            
            # Save the best model
            best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ New best model saved! Val Acc: {{avg_val_acc*100:.2f}}%")
            
            # Save checkpoint
            checkpoint_path = os.path.join(checkpoints_dir, "last_checkpoint.pth") 
            torch.save({{
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': avg_val_acc,
                'config': config
            }}, checkpoint_path)
        else:
            epochs_no_improve += 1
            print(f"No improvement for {{epochs_no_improve}} epochs")
            
            if epochs_no_improve >= patience:
                print(f"\\nEarly stopping after {{patience}} epochs without improvement.")
                break
    
    # Final evaluation with best model
    final_test_accuracy = evaluate_final_model(model, val_loader, device, checkpoints_dir, data_module, config)
    
    logger.close()
    print(f"\\n🎯 FINAL RESULTS - {{exp_name}}")
    print("=" * 60)
    print(f"Best Validation Accuracy: {{best_val_accuracy*100:.2f}}%")
    print(f"Final Test Accuracy: {{final_test_accuracy:.2f}}%") 
    print(f"Target Accuracy (Paper): {{config['target_accuracy']}}%")
    print(f"Difference from Target: {{final_test_accuracy - config['target_accuracy']:+.2f}}%")
    
    return final_test_accuracy

def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_num, logger, data_module, scaler):
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(train_loader, desc=f"Train Epoch {{epoch_num+1}}")
    for batch_idx, batch in enumerate(progress_bar):
        # Process batch through data module
        batch_dict = data_module.preprocess_batch(batch)
        data = batch_dict['inputs']
        target = batch_dict['labels']
        
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            output = model(data)
            loss = criterion(output, target)
        
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
        progress_bar.set_postfix(loss=f"{{current_loss:.4f}}", acc=f"{{current_acc:.4f}}")
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def validate_one_epoch(model, val_loader, criterion, device, epoch_num, logger, data_module):
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    
    progress_bar = tqdm(val_loader, desc=f"Val Epoch {{epoch_num+1}}")
    with torch.no_grad():
        for batch_idx, batch in enumerate(progress_bar):
            # Process batch through data module
            batch_dict = data_module.preprocess_batch(batch)
            data = batch_dict['inputs']
            target = batch_dict['labels']
            
            data, target = data.to(device), target.to(device)
            
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                output = model(data)
                loss = criterion(output, target)
            
            total_loss += loss.item() * data.size(0)
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
            
            current_loss = total_loss / total_samples
            current_acc = total_correct / total_samples
            progress_bar.set_postfix(loss=f"{{current_loss:.4f}}", acc=f"{{current_acc:.4f}}")
    
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

def evaluate_final_model(model, test_loader, device, checkpoints_dir, data_module, config):
    """Final evaluation with the best saved model."""
    print("\\n🔍 FINAL MODEL EVALUATION")
    print("-" * 40)
    
    # Load best model
    best_model_path = os.path.join(checkpoints_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        print(f"Loading best model from {{best_model_path}}")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: No saved model found, using current model weights")
    
    model.eval()
    total_correct = 0
    total_samples = 0
    
    print(f"Evaluating on {{len(test_loader)}} test batches...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation"):
            # Process batch through data module
            batch_dict = data_module.preprocess_batch(batch)
            data = batch_dict['inputs']
            target = batch_dict['labels']
            
            data, target = data.to(device), target.to(device)
            
            with autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                output = model(data)
            
            _, predicted = torch.max(output.data, 1)
            total_correct += (predicted == target).sum().item()
            total_samples += data.size(0)
    
    final_accuracy = (total_correct / total_samples) * 100
    
    print(f"✅ Final Test Accuracy: {{final_accuracy:.2f}}%")
    print(f"📊 Test Samples: {{total_samples:,}}")
    print(f"🎯 Target (Paper): {{config['target_accuracy']}}%")
    
    return final_accuracy

if __name__ == "__main__":
    final_acc = main()
'''
        
        # Write script to file
        script_path = self.log_dir / f"{exp_name}_train.py"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Created custom training script: {script_path}")
        return script_path
    
    def run_experiment(self, exp_name, config):
        """Run a single experiment."""
        print(f"🚀 STARTING: {exp_name}")
        print(f"📝 Description: {config['description']}")
        print(f"🎯 {config['expected_result']}")
        print("-" * 60)
        
        if self.dry_run:
            print("🔍 DRY RUN - Command would be executed")
            time.sleep(2)
            return True, "Dry run completed", 0.0
        
        start_time = time.time()
        
        try:
            # Create custom training script
            script_path = self.create_custom_train_script(exp_name, config)
            
            # Run the custom training script
            result = subprocess.run(
                ["python", str(script_path)],
                capture_output=False,  # Allow real-time output
                text=True,
                cwd=os.getcwd(),
                check=False
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {exp_name}")
                
                # Try to extract final accuracy from logs (simplified approach)
                final_accuracy = self.extract_final_accuracy(exp_name)
                
                return True, f"Completed successfully in {duration/3600:.1f} hours", final_accuracy
            else:
                print(f"❌ FAILED: {exp_name} (exit code: {result.returncode})")
                return False, f"Failed with exit code {result.returncode}", 0.0
                
        except Exception as e:
            print(f"❌ ERROR: {exp_name} - {str(e)}")
            return False, str(e), 0.0
        
        finally:
            print("-" * 60)
            print()
    
    def extract_final_accuracy(self, exp_name):
        """Try to extract final test accuracy from experiment logs (simplified)."""
        try:
            # This is a simplified approach - in a real scenario you might
            # want to parse tensorboard logs or save results to a JSON file
            log_path = self.log_dir / exp_name / "config.txt"
            if log_path.exists():
                # For now, return a placeholder - the actual accuracy would be
                # printed during execution and could be captured differently
                return 0.0
        except Exception:
            pass
        return 0.0
    
    def run_all_experiments(self):
        """Run all augmentation experiments."""
        self.print_header()
        self.print_experiment_plan()
        
        if not self.dry_run:
            print("⚠️  WARNING: ModelNet40 experiments may take several hours each!")
            response = input("🤔 Proceed with execution? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Execution cancelled by user.")
                return
        
        print("🎬 STARTING MODELNET40 AUGMENTATION STUDY")
        print("=" * 80)
        
        # Sort experiments by priority
        sorted_experiments = sorted(
            self.experiments.items(),
            key=lambda x: x[1]['priority']
        )
        
        successful = 0
        failed = 0
        results_summary = []
        
        for exp_name, config in sorted_experiments:
            success, message, final_accuracy = self.run_experiment(exp_name, config)
            
            self.results[exp_name] = {
                "success": success,
                "message": message,
                "final_accuracy": final_accuracy,
                "augmentation": config["augmentation"],
                "timestamp": datetime.datetime.now().isoformat(),
                "config": config
            }
            
            results_summary.append({
                "name": exp_name,
                "success": success,
                "accuracy": final_accuracy,
                "augmentation": config["augmentation"]
            })
            
            if success:
                successful += 1
                print(f"✅ {exp_name}: {message}")
                if final_accuracy > 0:
                    print(f"🎯 Final Test Accuracy: {final_accuracy:.2f}%")
            else:
                failed += 1
                print(f"❌ {exp_name}: {message}")
                
                if not self.dry_run:
                    response = input("🤔 Continue with remaining experiments? [Y/n]: ")
                    if response.lower() == 'n':
                        print("⏹️  Execution stopped by user.")
                        break
            
            print()
        
        # Generate final report
        self.generate_final_report(successful, failed, results_summary)
    
    def generate_final_report(self, successful, failed, results_summary):
        """Generate final execution report with ModelNet40 results comparison."""
        end_time = datetime.datetime.now()
        total_duration = end_time - self.start_time
        
        print("=" * 80)
        print("📊 MODELNET40 AUGMENTATION STUDY - FINAL REPORT")
        print("=" * 80)
        print(f"🕐 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕑 End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total Duration: {str(total_duration)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        if successful + failed > 0:
            print(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
        print()
        
        # Results comparison table
        print("📋 AUGMENTATION STUDY RESULTS:")
        print("=" * 80)
        print(f"{'Experiment':<35} {'Augmentation':<20} {'Top-1 Acc':<10} {'vs Target':<10}")
        print("-" * 80)
        
        target_acc = self.paper_config['target_accuracy']
        
        for result in results_summary:
            if result['success'] and result['accuracy'] > 0:
                diff = result['accuracy'] - target_acc
                diff_str = f"{diff:+.1f}%"
                acc_str = f"{result['accuracy']:.1f}%"
            else:
                acc_str = "FAILED"
                diff_str = "N/A"
            
            print(f"{result['name']:<35} {result['augmentation']:<20} {acc_str:<10} {diff_str:<10}")
        
        print("-" * 80)
        print(f"{'TARGET (Paper)':<35} {'scale_only':<20} {f'{target_acc}%':<10} {'baseline':<10}")
        print("=" * 80)
        print()
        
        # Analysis summary
        print("🔍 ANALYSIS SUMMARY:")
        print("-" * 50)
        
        baseline_found = False
        best_acc = 0
        worst_acc = 100
        
        for result in results_summary:
            if result['success'] and result['accuracy'] > 0:
                if result['augmentation'] == 'scale_only':
                    baseline_found = True
                    baseline_acc = result['accuracy']
                    print(f"📌 Baseline (scale only): {baseline_acc:.1f}% (target: {target_acc}%)")
                    if abs(baseline_acc - target_acc) < 2.0:
                        print("   ✅ Successfully reproduced paper results!")
                    else:
                        print(f"   ⚠️  Difference from paper: {baseline_acc - target_acc:+.1f}%")
                
                best_acc = max(best_acc, result['accuracy'])
                worst_acc = min(worst_acc, result['accuracy'])
        
        if successful > 1:
            print(f"📈 Best accuracy: {best_acc:.1f}%")
            print(f"📉 Worst accuracy: {worst_acc:.1f}%")
            print(f"📊 Accuracy range: {best_acc - worst_acc:.1f}%")
        
        print()
        
        # Conclusions
        print("💡 CONCLUSIONS:")
        print("-" * 30)
        translation_worse = False
        rotation_worse = False
        
        for result in results_summary:
            if result['success'] and result['accuracy'] > 0:
                if result['augmentation'] == 'scale_translation' and baseline_found:
                    if result['accuracy'] < baseline_acc:
                        translation_worse = True
                        print(f"• Translation augmentation decreased accuracy by {baseline_acc - result['accuracy']:.1f}%")
                    else:
                        print(f"• Translation augmentation improved accuracy by {result['accuracy'] - baseline_acc:+.1f}%")
                
                elif result['augmentation'] == 'scale_rotation' and baseline_found:
                    if result['accuracy'] < baseline_acc:
                        rotation_worse = True
                        print(f"• Rotation augmentation decreased accuracy by {baseline_acc - result['accuracy']:.1f}%")
                    else:
                        print(f"• Rotation augmentation improved accuracy by {result['accuracy'] - baseline_acc:+.1f}%")
        
        if translation_worse and rotation_worse:
            print("• ✅ Paper conclusion confirmed: additional augmentations don't improve performance")
        elif not translation_worse or not rotation_worse:
            print("• 🤔 Results differ from paper - additional augmentations may help in some cases")
        
        print()
        
        # Save detailed report
        report_path = self.log_dir / "modelnet40_augmentation_study_report.json"
        report_data = {
            "study_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "total_duration_seconds": total_duration.total_seconds(),
                "successful_experiments": successful,
                "failed_experiments": failed
            },
            "paper_target": {
                "accuracy": self.paper_config['target_accuracy'],
                "configuration": self.paper_config
            },
            "results": self.results,
            "summary": results_summary
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"💾 Detailed report saved to: {report_path}")
        print("=" * 80)
    
    def run_specific_experiment(self, experiment_name):
        """Run only a specific experiment."""
        if experiment_name not in self.experiments:
            print(f"❌ Unknown experiment: {experiment_name}")
            print(f"Available experiments: {list(self.experiments.keys())}")
            return
            
        print(f"🎯 RUNNING SPECIFIC EXPERIMENT: {experiment_name}")
        config = self.experiments[experiment_name]
        success, message, final_accuracy = self.run_experiment(experiment_name, config)
        
        self.results[experiment_name] = {
            "success": success,
            "message": message,
            "final_accuracy": final_accuracy,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        print(f"\n🎯 EXPERIMENT COMPLETED: {experiment_name}")
        if success and final_accuracy > 0:
            print(f"Final Test Accuracy: {final_accuracy:.2f}%")
            print(f"Target Accuracy: {self.paper_config['target_accuracy']}%")
            print(f"Difference: {final_accuracy - self.paper_config['target_accuracy']:+.2f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ModelNet40 augmentation study")
    parser.add_argument('--dry-run', action='store_true',
                       help='Print commands without executing')
    parser.add_argument('--experiment', type=str,
                       help='Run specific experiment only')
    parser.add_argument('--log-dir', default='logs',
                       help='Directory for saving logs')
    
    args = parser.parse_args()
    
    runner = ModelNet40AugmentationStudy(
        log_dir=args.log_dir,
        dry_run=args.dry_run
    )
    
    if args.experiment:
        # Run specific experiment
        runner.run_specific_experiment(args.experiment)
    else:
        # Run all experiments
        runner.run_all_experiments()


if __name__ == "__main__":
    main()
