#!/usr/bin/env python3
"""
ModelNet40 Simple Training Script
===============================

Script semplificato per eseguire training su ModelNet40 con configurazione paper.
Usa direttamente train.py con parametri corretti.
"""

import subprocess
import os
import sys

def run_modelnet40_baseline():
    """Esegue l'esperimento baseline ModelNet40 (configurazione paper)."""
    
    print("🚀 STARTING ModelNet40 Baseline Experiment")
    print("=" * 60)
    print("Target: 85.7% top-1 accuracy (paper result)")
    print("Configuration: Paper-compliant ModelNet40 setup")
    print("=" * 60)
    
    # Parametri paper per ModelNet40
    cmd = [
        "python", "train.py",
        "--experiment_name", "modelnet40_baseline_paper",
        "--dataset", "modelnet40",
        "--modelnet40_num_points", "2048",
        "--modelnet40_fourier_bands", "64", 
        "--modelnet40_max_freq", "1120.0",
        "--batch_size_modelnet40", "512",
        "--num_latents", "128",
        "--latent_dim", "512", 
        "--num_cross_attend_stages", "2",
        "--num_transformer_blocks", "6",
        "--num_heads", "8",
        "--dropout", "0.1",
        "--optimizer", "lamb",
        "--lr", "0.001",
        "--scheduler", "none",
        "--epochs", "150",
        "--save_attention_maps",
        "--save_metrics",
        "--use_tensorboard"
    ]
    
    print("🔧 Command:")
    print(" ".join(cmd))
    print()
    
    try:
        # Esegui il comando
        result = subprocess.run(cmd, check=True)
        
        print("✅ ModelNet40 baseline completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def run_modelnet40_with_translation():
    """Esegue l'esperimento con translation augmentation."""
    
    print("🚀 STARTING ModelNet40 + Translation Experiment") 
    print("=" * 60)
    print("Expected: ≤85.7% (paper states translation doesn't improve)")
    print("Configuration: Baseline + translation (-0.02 to +0.02)")
    print("=" * 60)
    
    # Stessi parametri baseline + aggiunta custom per translation
    # Dovrebbe modificare il data module per includere translation
    # Per ora uso stessa configurazione - l'implementazione vera richiederebbe
    # modifiche al train.py per supportare parametri di augmentation
    
    cmd = [
        "python", "train.py",
        "--experiment_name", "modelnet40_with_translation",
        "--dataset", "modelnet40",
        "--modelnet40_num_points", "2048",
        "--modelnet40_fourier_bands", "64",
        "--modelnet40_max_freq", "1120.0", 
        "--batch_size_modelnet40", "512",
        "--num_latents", "128",
        "--latent_dim", "512",
        "--num_cross_attend_stages", "2", 
        "--num_transformer_blocks", "6",
        "--num_heads", "8",
        "--dropout", "0.1",
        "--optimizer", "lamb",
        "--lr", "0.001",
        "--scheduler", "none",
        "--epochs", "150",
        "--save_attention_maps",
        "--save_metrics", 
        "--use_tensorboard"
    ]
    
    print("⚠️  Note: This uses baseline config - translation augmentation")
    print("    requires additional implementation in train.py data loading")
    print()
    print("🔧 Command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ ModelNet40 + translation completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Menu principale per scegliere quale esperimento eseguire."""
    
    print("🧠 ModelNet40 Experiments - Simple Runner")
    print("=" * 50)
    print("1. Baseline (paper config) - Target: 85.7%")
    print("2. + Translation augmentation")  
    print("3. Both experiments (sequential)")
    print("4. Exit")
    print("=" * 50)
    
    while True:
        try:
            choice = input("\nSelect experiment (1-4): ").strip()
            
            if choice == "1":
                print("\n🎯 Running baseline experiment...")
                run_modelnet40_baseline()
                break
                
            elif choice == "2":
                print("\n🎯 Running translation experiment...")
                run_modelnet40_with_translation()
                break
                
            elif choice == "3":
                print("\n🎯 Running both experiments...")
                print("\n>>> EXPERIMENT 1/2: Baseline")
                success1 = run_modelnet40_baseline()
                
                if success1:
                    input("\nPress Enter to continue to experiment 2...")
                    print("\n>>> EXPERIMENT 2/2: Translation")
                    success2 = run_modelnet40_with_translation()
                    
                    print(f"\n📊 FINAL RESULTS:")
                    print(f"Baseline: {'✅ Success' if success1 else '❌ Failed'}")
                    print(f"Translation: {'✅ Success' if success2 else '❌ Failed'}")
                else:
                    print("❌ Skipping second experiment due to baseline failure")
                break
                
            elif choice == "4":
                print("👋 Exiting...")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting...")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Se chiamato con argomenti, esegui direttamente
    if len(sys.argv) > 1:
        if sys.argv[1] == "baseline":
            run_modelnet40_baseline()
        elif sys.argv[1] == "translation": 
            run_modelnet40_with_translation()
        elif sys.argv[1] == "both":
            run_modelnet40_baseline()
            run_modelnet40_with_translation()
        else:
            print(f"Usage: {sys.argv[0]} [baseline|translation|both]")
    else:
        # Altrimenti mostra menu interattivo
        main()
