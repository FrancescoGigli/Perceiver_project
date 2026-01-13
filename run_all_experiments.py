#!/usr/bin/env python3
"""
Script per eseguire tutti i 6 esperimenti Perceiver in sequenza.

Questo script:
1. Esegue tutti i 6 esperimenti con le configurazioni ottimali
2. Monitora il progresso e salva logs
3. Gestisce errori e recovery
4. Genera report finale con tutti i risultati
"""

import subprocess
import time
import os
import json
import datetime
from pathlib import Path

class PerceiverExperimentRunner:
    def __init__(self, log_dir="logs", dry_run=False):
        """
        Initialize experiment runner.
        
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
        
        # Experiment configurations
        self.experiments = {
            "exp1_baseline_fourier": {
                "description": "Baseline con Fourier PE - Riferimento principale",
                "priority": 1,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "exp1_baseline_fourier",
                    "--dataset", "cifar10",
                    "--cifar10_fourier_bands", "64",
                    "--cifar10_max_freq", "32.0",
                    "--num_latents", "96",
                    "--latent_dim", "384",
                    "--num_cross_attend_stages", "4",
                    "--num_transformer_blocks", "4",
                    "--num_heads", "3",
                    "--dropout", "0.2",
                    "--optimizer", "lamb",
                    "--lr", "0.004",
                    "--scheduler", "multistep",
                    "--epochs", "120",
                    "--batch_size_cifar10", "64",
                    "--save_attention_maps",
                    "--attention_save_interval", "10",
                    "--use_tensorboard"
                ]
            },
            
            "exp3A_fourier_control": {
                "description": "Fourier Control - Identico a Exp1 per confronto con RGB-only",
                "priority": 2,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "exp3A_fourier_control",
                    "--dataset", "cifar10",
                    "--cifar10_fourier_bands", "64",
                    "--cifar10_max_freq", "32.0",
                    "--num_latents", "96",
                    "--latent_dim", "384",
                    "--num_cross_attend_stages", "4",
                    "--num_transformer_blocks", "4",
                    "--num_heads", "3",
                    "--dropout", "0.2",
                    "--optimizer", "lamb",
                    "--lr", "0.004",
                    "--scheduler", "multistep",
                    "--epochs", "120",
                    "--batch_size_cifar10", "64",
                    "--save_attention_maps",
                    "--use_tensorboard"
                ]
            },
            
            "exp3B_rgb_only": {
                "description": "RGB-only (NO PE) - Test importanza positional encoding",
                "priority": 3,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "exp3B_rgb_only",
                    "--dataset", "cifar10",
                    "--no_positional_encoding",
                    "--num_latents", "96",
                    "--latent_dim", "384",
                    "--num_cross_attend_stages", "4",
                    "--num_transformer_blocks", "4",
                    "--num_heads", "3",
                    "--dropout", "0.2",
                    "--optimizer", "lamb",
                    "--lr", "0.004",
                    "--scheduler", "multistep",
                    "--epochs", "120",
                    "--batch_size_cifar10", "64",
                    "--save_attention_maps",
                    "--use_tensorboard"
                ]
            },
            
            "exp4A_weight_sharing_control": {
                "description": "Weight Sharing Control - Baseline per confronto No Weight Sharing",
                "priority": 4,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "exp4A_weight_sharing_control",
                    "--dataset", "cifar10",
                    "--cifar10_fourier_bands", "64",
                    "--cifar10_max_freq", "32.0",
                    "--num_latents", "96",
                    "--latent_dim", "384",
                    "--num_cross_attend_stages", "4",
                    "--num_transformer_blocks", "4",
                    "--num_heads", "3",
                    "--dropout", "0.2",
                    "--optimizer", "lamb",
                    "--lr", "0.004",
                    "--scheduler", "multistep",
                    "--epochs", "120",
                    "--batch_size_cifar10", "64",
                    "--save_attention_maps",
                    "--use_tensorboard"
                ]
            },
            
            "exp4B_no_weight_sharing": {
                "description": "No Weight Sharing - Test overfitting e parameter efficiency",
                "priority": 5,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "exp4B_no_weight_sharing",
                    "--dataset", "cifar10",
                    "--cifar10_fourier_bands", "64",
                    "--cifar10_max_freq", "32.0",
                    "--num_latents", "96",
                    "--latent_dim", "384",
                    "--num_cross_attend_stages", "4",
                    "--num_transformer_blocks", "4",
                    "--num_heads", "3",
                    "--dropout", "0.2",
                    "--no_weight_sharing",
                    "--optimizer", "lamb",
                    "--lr", "0.004",
                    "--scheduler", "multistep",
                    "--epochs", "120",
                    "--batch_size_cifar10", "64",
                    "--save_attention_maps",
                    "--use_tensorboard"
                ]
            },
            
            "exp6_fourier_permuted": {
                "description": "Fourier PE + Permutazione - Test robustezza spaziale Fourier",
                "priority": 6,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "exp6_fourier_permuted",
                    "--dataset", "cifar10",
                    "--permute_pixels",
                    "--permute_pixels_seed", "42",
                    "--cifar10_fourier_bands", "64",
                    "--cifar10_max_freq", "32.0",
                    "--num_latents", "96",
                    "--latent_dim", "384",
                    "--num_cross_attend_stages", "4",
                    "--num_transformer_blocks", "4",
                    "--num_heads", "3",
                    "--dropout", "0.2",
                    "--optimizer", "lamb",
                    "--lr", "0.004",
                    "--scheduler", "multistep",
                    "--epochs", "120",
                    "--batch_size_cifar10", "64",
                    "--save_attention_maps",
                    "--attention_save_interval", "10",
                    "--save_metrics",
                    "--use_tensorboard"
                ]
            },
            
            "exp2_learned_pe_permuted": {
                "description": "Learned PE + Permutazione - Robustezza learned PE (più complesso)",
                "priority": 7,  # Ultimo perché più complesso
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "exp2_learned_pe_permuted",
                    "--dataset", "cifar10",
                    "--use_learned_pe",
                    "--permute_pixels",
                    "--permute_pixels_seed", "42",
                    "--num_latents", "96",
                    "--latent_dim", "384",
                    "--num_cross_attend_stages", "1",  # Ridotto per learned PE
                    "--num_transformer_blocks", "4",
                    "--num_heads", "3",
                    "--dropout", "0.2",
                    "--optimizer", "lamb",
                    "--lr", "0.004",
                    "--scheduler", "multistep",
                    "--epochs", "120",
                    "--batch_size_cifar10", "64",
                    "--save_attention_maps",
                    "--attention_save_interval", "10",
                    "--save_metrics",
                    "--use_tensorboard"
                ]
            },
            
            # ======= MODELNET40 EXPERIMENTS =======
            
            "modelnet40_baseline": {
                "description": "ModelNet40 Baseline - Lightweight config (target ~80-85%)",
                "priority": 8,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "modelnet40_baseline_lightweight",
                    "--dataset", "modelnet40",
                    "--modelnet40_num_points", "1024",  # Ridotto da 2048 per memoria
                    "--modelnet40_fourier_bands", "32", # Ridotto da 64 per memoria
                    "--modelnet40_max_freq", "1120.0",  # Mantiene freq paper
                    "--batch_size_modelnet40", "64",    # Ultra ridotto per memoria GPU
                    "--num_latents", "96",              # Ridotto da 128 
                    "--latent_dim", "384",              # Ridotto da 512
                    "--num_cross_attend_stages", "2",   # Mantiene paper
                    "--num_transformer_blocks", "4",    # Ridotto da 6 
                    "--num_heads", "6",                 # Ridotto da 8
                    "--dropout", "0.1",
                    "--optimizer", "lamb",              # LAMB (paper)
                    "--lr", "0.001",                    # LR costante (paper)
                    "--scheduler", "none",              # LR costante (paper)
                    "--epochs", "250",                  # Più epoche per compensare
                    "--save_attention_maps",
                    "--save_metrics",
                    "--use_tensorboard"
                ]
            },
            
            "modelnet40_with_translation": {
                "description": "ModelNet40 + Translation - Lightweight test (paper: non migliora)",
                "priority": 9,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "modelnet40_with_translation_lightweight",
                    "--dataset", "modelnet40",
                    "--modelnet40_num_points", "1024", # Ridotto per memoria
                    "--modelnet40_fourier_bands", "32", # Ridotto per memoria
                    "--modelnet40_max_freq", "1120.0",
                    "--batch_size_modelnet40", "64",   # Ultra ridotto per memoria GPU
                    "--num_latents", "96",             # Ridotto da 128
                    "--latent_dim", "384",             # Ridotto da 512
                    "--num_cross_attend_stages", "2",
                    "--num_transformer_blocks", "4",   # Ridotto da 6
                    "--num_heads", "6",                # Ridotto da 8
                    "--dropout", "0.1",
                    "--optimizer", "lamb",
                    "--lr", "0.001",
                    "--scheduler", "none",
                    "--epochs", "250",                 # Più epoche per compensare
                    "--save_attention_maps",
                    "--save_metrics",
                    "--use_tensorboard"
                ]
            },
            
            "modelnet40_with_rotation": {
                "description": "ModelNet40 + Rotation - Lightweight test (paper: non migliora)",
                "priority": 10,
                "cmd": [
                    "python", "train.py",
                    "--experiment_name", "modelnet40_with_rotation_lightweight",
                    "--dataset", "modelnet40",
                    "--modelnet40_num_points", "1024", # Ridotto per memoria
                    "--modelnet40_fourier_bands", "32", # Ridotto per memoria
                    "--modelnet40_max_freq", "1120.0",
                    "--batch_size_modelnet40", "64",   # Ultra ridotto per memoria GPU
                    "--num_latents", "96",             # Ridotto da 128
                    "--latent_dim", "384",             # Ridotto da 512
                    "--num_cross_attend_stages", "2",
                    "--num_transformer_blocks", "4",   # Ridotto da 6
                    "--num_heads", "6",                # Ridotto da 8
                    "--dropout", "0.1",
                    "--optimizer", "lamb",
                    "--lr", "0.001",
                    "--scheduler", "none", 
                    "--epochs", "250",                 # Più epoche per compensare
                    "--save_attention_maps",
                    "--save_metrics",
                    "--use_tensorboard"
                ]
            }
        }
    
    def print_header(self):
        """Print script header and info."""
        print("=" * 80)
        print("🧠 PERCEIVER EXPERIMENTS - AUTOMATED RUNNER")
        print("=" * 80)
        print(f"📅 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📂 Log Directory: {self.log_dir}")
        print(f"🧪 Total Experiments: {len(self.experiments)}")
        print(f"🖥️  Mode: {'DRY RUN' if self.dry_run else 'EXECUTION'}")
        print("=" * 80)
        print()
    
    def print_experiment_plan(self):
        """Print the execution plan."""
        print("📋 EXECUTION PLAN:")
        print("-" * 50)
        
        # Sort by priority
        sorted_exps = sorted(self.experiments.items(), key=lambda x: x[1]['priority'])
        
        for i, (exp_name, config) in enumerate(sorted_exps, 1):
            print(f"{i}. {exp_name}")
            print(f"   📝 {config['description']}")
            print(f"   🔧 Key flags: {self._get_key_flags(config['cmd'])}")
            print()
        print("-" * 50)
        print()
    
    def _get_key_flags(self, cmd):
        """Extract key flags from command."""
        key_flags = []
        if "--no_positional_encoding" in cmd:
            key_flags.append("no_PE")
        if "--use_learned_pe" in cmd:
            key_flags.append("learned_PE")
        if "--permute_pixels" in cmd:
            key_flags.append("permuted")
        if "--no_weight_sharing" in cmd:
            key_flags.append("no_sharing")
        if "--cifar10_fourier_bands" in cmd:
            bands_idx = cmd.index("--cifar10_fourier_bands") + 1
            key_flags.append(f"fourier_{cmd[bands_idx]}")
            
        return ", ".join(key_flags) if key_flags else "standard"
    
    def run_experiment(self, exp_name, config):
        """Run a single experiment."""
        print(f"🚀 STARTING: {exp_name}")
        print(f"📝 Description: {config['description']}")
        print(f"🔧 Command: {' '.join(config['cmd'])}")
        print("-" * 60)
        
        if self.dry_run:
            print("🔍 DRY RUN - Command would be executed")
            time.sleep(2)  # Simulate some execution time
            return True, "Dry run completed"
        
        start_time = time.time()
        
        try:
            # Run the training command
            result = subprocess.run(
                config['cmd'],
                capture_output=False,  # Allow real-time output
                text=True,
                cwd=os.getcwd(),
                check=False  # Don't raise exception on non-zero return
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ SUCCESS: {exp_name}")
                return True, f"Completed successfully"
            else:
                print(f"❌ FAILED: {exp_name} (exit code: {result.returncode})")
                return False, f"Failed with exit code {result.returncode}"
                
        except Exception as e:
            print(f"❌ ERROR: {exp_name} - {str(e)}")
            return False, str(e)
        
        finally:
            print("-" * 60)
            print()
    
    def run_all_experiments(self):
        """Run all experiments in priority order."""
        self.print_header()
        self.print_experiment_plan()
        
        if not self.dry_run:
            response = input("🤔 Proceed with execution? [y/N]: ")
            if response.lower() != 'y':
                print("❌ Execution cancelled by user.")
                return
        
        print("🎬 STARTING EXPERIMENT EXECUTION")
        print("=" * 80)
        
        # Sort experiments by priority
        sorted_experiments = sorted(
            self.experiments.items(), 
            key=lambda x: x[1]['priority']
        )
        
        successful = 0
        failed = 0
        
        for exp_name, config in sorted_experiments:
            success, message = self.run_experiment(exp_name, config)
            
            self.results[exp_name] = {
                "success": success,
                "message": message,
                "timestamp": datetime.datetime.now().isoformat(),
                "config": config
            }
            
            if success:
                successful += 1
                print(f"✅ {exp_name}: {message}")
            else:
                failed += 1
                print(f"❌ {exp_name}: {message}")
                
                # Ask if should continue
                if not self.dry_run:
                    response = input("🤔 Continue with remaining experiments? [Y/n]: ")
                    if response.lower() == 'n':
                        print("⏹️  Execution stopped by user.")
                        break
            
            print()
        
        # Generate final report
        self.generate_final_report(successful, failed)
    
    def generate_final_report(self, successful, failed):
        """Generate final execution report."""
        end_time = datetime.datetime.now()
        total_duration = end_time - self.start_time
        
        print("=" * 80)
        print("📊 FINAL EXECUTION REPORT")
        print("=" * 80)
        print(f"🕐 Start Time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🕑 End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Total Duration: {str(total_duration)}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"📊 Success Rate: {successful/(successful+failed)*100:.1f}%")
        print()
        
        # Detailed results
        print("📋 DETAILED RESULTS:")
        print("-" * 50)
        for exp_name, result in self.results.items():
            status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
            print(f"{status}: {exp_name}")
            print(f"   📝 {result['message']}")
            print(f"   🕐 {result['timestamp']}")
            print()
        
        # Next steps
        if successful > 0:
            print("🎯 NEXT STEPS:")
            print("-" * 30)
            print("1. Check experiment results in logs/ directory")
            print("2. Run attention analysis:")
            print("   python analyze_attention_evolution.py \\")
            print("          --logs_dir logs \\")
            print("          --create_evolution \\")
            print("          --create_comparative_analysis \\")
            print("          --save_report")
            print("3. Generate standard visualizations:")
            print("   python visualize_results.py --output_dir logs/exp1_baseline_fourier")
            print("   python visualize_attention.py --logs_dir logs")
            print()
        
        # Save JSON report
        report_path = self.log_dir / "execution_report.json"
        report_data = {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "total_duration_seconds": total_duration.total_seconds(),
            "successful": successful,
            "failed": failed,
            "experiments": self.results
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"💾 Detailed report saved to: {report_path}")
        print("=" * 80)
    
    def run_specific_experiments(self, experiment_names):
        """Run only specific experiments."""
        print(f"🎯 RUNNING SPECIFIC EXPERIMENTS: {experiment_names}")
        
        for exp_name in experiment_names:
            if exp_name not in self.experiments:
                print(f"❌ Unknown experiment: {exp_name}")
                print(f"Available: {list(self.experiments.keys())}")
                continue
                
            config = self.experiments[exp_name]
            success, message = self.run_experiment(exp_name, config)
            
            self.results[exp_name] = {
                "success": success,
                "message": message,
                "timestamp": datetime.datetime.now().isoformat()
            }
    
    def run_post_training_analysis(self):
        """Run post-training analysis (Experiment 5)."""
        print("🔍 RUNNING POST-TRAINING ANALYSIS (Experiment 5)")
        print("=" * 60)
        
        analysis_cmd = [
            "python", "analyze_attention_evolution.py",
            "--logs_dir", "logs",
            "--create_evolution",
            "--create_comparative_analysis", 
            "--save_report",
            "--output_dir", "attention_analysis"
        ]
        
        print(f"🔧 Command: {' '.join(analysis_cmd)}")
        
        if self.dry_run:
            print("🔍 DRY RUN - Analysis command would be executed")
            return True
        
        try:
            result = subprocess.run(analysis_cmd, check=True)
            print("✅ Attention analysis completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Analysis failed: {e}")
            return False
        except Exception as e:
            print(f"❌ Analysis error: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Perceiver experiments")
    parser.add_argument('--dry-run', action='store_true', 
                       help='Print commands without executing')
    parser.add_argument('--experiments', nargs='+',
                       help='Specific experiments to run')
    parser.add_argument('--analysis-only', action='store_true',
                       help='Run only post-training analysis')
    parser.add_argument('--log-dir', default='logs',
                       help='Directory for saving logs')
    
    args = parser.parse_args()
    
    runner = PerceiverExperimentRunner(
        log_dir=args.log_dir,
        dry_run=args.dry_run
    )
    
    if args.analysis_only:
        # Run only post-training analysis
        runner.run_post_training_analysis()
    elif args.experiments:
        # Run specific experiments
        runner.run_specific_experiments(args.experiments)
    else:
        # Run all experiments
        runner.run_all_experiments()
        
        # Ask if should run analysis
        if not args.dry_run and runner.results:
            print()
            response = input("🔍 Run post-training attention analysis now? [Y/n]: ")
            if response.lower() != 'n':
                runner.run_post_training_analysis()

if __name__ == "__main__":
    main()
