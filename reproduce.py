# reproduce.py
# Script interattivo per riprodurre gli esperimenti del progetto Perceiver.
# Ogni esperimento lancia train.py con i parametri esatti usati nei risultati originali.
#
# Uso:
#   python reproduce.py            → menu interattivo
#   python reproduce.py --list     → mostra tutti gli esperimenti
#   python reproduce.py --run 1    → lancia esperimento #1 direttamente

import subprocess
import sys
import argparse
import time

# ─────────────────────────────────────────────────────────────────────────────
# Definizione esperimenti
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = {
    # ── CIFAR-10 Ablation Study (Perceiver) ──────────────────────────────────
    1: {
        "name": "exp1_baseline_fourier",
        "description": "CIFAR-10 Baseline con Fourier PE (riferimento principale)",
        "category": "CIFAR-10",
        "estimated_time": "~3 ore (GPU)",
        "cmd": [
            "--experiment_name", "exp1_baseline_fourier",
            "--dataset", "cifar10",
            "--cifar10_fourier_bands", "64", "--cifar10_max_freq", "32.0",
            "--num_latents", "96", "--latent_dim", "384",
            "--num_cross_attend_stages", "4", "--num_transformer_blocks", "4",
            "--num_heads", "3", "--dropout", "0.2",
            "--optimizer", "lamb", "--lr", "0.004",
            "--scheduler", "multistep", "--epochs", "120",
            "--batch_size_cifar10", "64",
            "--save_attention_maps", "--attention_save_interval", "10",
            "--use_tensorboard",
        ],
    },
    2: {
        "name": "exp3B_rgb_only",
        "description": "CIFAR-10 RGB Only — senza positional encoding",
        "category": "CIFAR-10",
        "estimated_time": "~3 ore (GPU)",
        "cmd": [
            "--experiment_name", "exp3B_rgb_only",
            "--dataset", "cifar10", "--no_positional_encoding",
            "--num_latents", "96", "--latent_dim", "384",
            "--num_cross_attend_stages", "4", "--num_transformer_blocks", "4",
            "--num_heads", "3", "--dropout", "0.2",
            "--optimizer", "lamb", "--lr", "0.004",
            "--scheduler", "multistep", "--epochs", "120",
            "--batch_size_cifar10", "64",
            "--save_attention_maps", "--use_tensorboard",
        ],
    },
    3: {
        "name": "exp4B_no_weight_sharing",
        "description": "CIFAR-10 senza Weight Sharing (più parametri)",
        "category": "CIFAR-10",
        "estimated_time": "~5 ore (GPU)",
        "cmd": [
            "--experiment_name", "exp4B_no_weight_sharing",
            "--dataset", "cifar10",
            "--cifar10_fourier_bands", "64", "--cifar10_max_freq", "32.0",
            "--num_latents", "96", "--latent_dim", "384",
            "--num_cross_attend_stages", "4", "--num_transformer_blocks", "4",
            "--num_heads", "3", "--dropout", "0.2", "--no_weight_sharing",
            "--optimizer", "lamb", "--lr", "0.004",
            "--scheduler", "multistep", "--epochs", "120",
            "--batch_size_cifar10", "64",
            "--save_attention_maps", "--use_tensorboard",
        ],
    },
    4: {
        "name": "exp6_fourier_permuted",
        "description": "CIFAR-10 Fourier PE + Pixel Permutati (robustezza spaziale)",
        "category": "CIFAR-10",
        "estimated_time": "~4 ore (GPU)",
        "cmd": [
            "--experiment_name", "exp6_fourier_permuted",
            "--dataset", "cifar10",
            "--permute_pixels", "--permute_pixels_seed", "42",
            "--cifar10_fourier_bands", "64", "--cifar10_max_freq", "32.0",
            "--num_latents", "96", "--latent_dim", "384",
            "--num_cross_attend_stages", "4", "--num_transformer_blocks", "4",
            "--num_heads", "3", "--dropout", "0.2",
            "--optimizer", "lamb", "--lr", "0.004",
            "--scheduler", "multistep", "--epochs", "120",
            "--batch_size_cifar10", "64",
            "--save_attention_maps", "--attention_save_interval", "10",
            "--save_metrics", "--use_tensorboard",
        ],
    },
    # ── CIFAR-10 Perceiver IO ────────────────────────────────────────────────
    5: {
        "name": "exp_cifar10_perceiver_io",
        "description": "CIFAR-10 con Perceiver IO (decoder con output queries)",
        "category": "CIFAR-10",
        "estimated_time": "~4 ore (GPU)",
        "cmd": [
            "--experiment_name", "exp_cifar10_perceiver_io",
            "--dataset", "cifar10",
            "--model_type", "perceiver_io", "--model_task", "classification",
            "--cifar10_fourier_bands", "64", "--cifar10_max_freq", "32.0",
            "--num_latents", "128", "--latent_dim", "512",
            "--num_cross_attend_stages", "1", "--num_transformer_blocks", "4",
            "--num_heads", "8", "--dropout", "0.1",
            "--num_output_queries", "1",
            "--optimizer", "lamb", "--lr", "0.001",
            "--scheduler", "multistep", "--epochs", "120",
            "--batch_size_cifar10", "64",
            "--save_attention_maps", "--use_tensorboard",
        ],
    },
    # ── ModelNet40 ───────────────────────────────────────────────────────────
    6: {
        "name": "modelnet40_baseline_paper",
        "description": "ModelNet40 Baseline (configurazione paper)",
        "category": "ModelNet40",
        "estimated_time": "~12 ore (GPU)",
        "cmd": [
            "--experiment_name", "modelnet40_baseline_paper",
            "--dataset", "modelnet40",
            "--modelnet40_num_points", "2048",
            "--modelnet40_fourier_bands", "64", "--modelnet40_max_freq", "1120.0",
            "--batch_size_modelnet40", "128",
            "--num_latents", "128", "--latent_dim", "512",
            "--num_cross_attend_stages", "2", "--num_transformer_blocks", "6",
            "--num_heads", "8", "--dropout", "0.1",
            "--optimizer", "lamb", "--lr", "0.001",
            "--scheduler", "none", "--epochs", "200",
            "--save_attention_maps", "--save_metrics", "--use_tensorboard",
        ],
    },
    # ── WikiText-103 MLM Pre-training ────────────────────────────────────────
    7: {
        "name": "mlm_wikitext103_optimized_batch32",
        "description": "Pre-training MLM su WikiText-103 (Perceiver IO, byte-level)",
        "category": "WikiText",
        "estimated_time": "~24+ ore (GPU)",
        "cmd": [
            "--experiment_name", "mlm_wikitext103_optimized_batch32",
            "--dataset", "wikitext103",
            "--model_type", "perceiver_io", "--model_task", "mlm",
            "--num_latents", "128", "--latent_dim", "512",
            "--num_cross_attend_stages", "1", "--num_transformer_blocks", "4",
            "--num_heads", "8", "--dropout", "0.1",
            "--text_seq_len", "1024", "--text_fourier_dim", "64", "--text_max_freq", "64.0",
            "--mlm_mask_prob", "0.15", "--mlm_vocab_size", "256",
            "--num_output_queries", "1024",
            "--optimizer", "lamb", "--lr", "0.001",
            "--scheduler", "multistep", "--epochs", "50",
            "--batch_size_cifar10", "32", "--num_workers", "2",
        ],
    },
    # ── GLUE Fine-tuning ─────────────────────────────────────────────────────
    8: {
        "name": "glue_sst2_finetune",
        "description": "Fine-tuning SST-2 (sentiment analysis, da checkpoint MLM)",
        "category": "GLUE",
        "estimated_time": "~2 ore (GPU)",
        "cmd": [
            "--experiment_name", "glue_sst2_finetune",
            "--dataset", "glue_sst2",
            "--model_type", "perceiver_io", "--model_task", "classification",
            "--num_latents", "128", "--latent_dim", "512",
            "--num_cross_attend_stages", "1", "--num_transformer_blocks", "4",
            "--num_heads", "8", "--dropout", "0.1",
            "--num_output_queries", "2",
            "--text_seq_len", "512", "--text_fourier_dim", "64", "--text_max_freq", "64.0",
            "--optimizer", "lamb", "--lr", "0.0005",
            "--scheduler", "multistep", "--epochs", "30",
            "--batch_size_cifar10", "32",
            "--load_checkpoint_path", "logs/mlm_wikitext103_optimized_batch32/checkpoints/last_checkpoint.pth",
        ],
    },
    9: {
        "name": "glue_all_tasks",
        "description": "Fine-tuning TUTTI i task GLUE (7 task in sequenza, da checkpoint MLM)",
        "category": "GLUE",
        "estimated_time": "~10+ ore (GPU)",
        "cmd": "__run_all_glue__",  # Flag speciale
    },
}


def print_menu():
    """Stampa il menu con tutti gli esperimenti disponibili."""
    print("\n" + "=" * 70)
    print("  PERCEIVER PROJECT — Riproduzione Esperimenti")
    print("=" * 70)

    current_category = None
    for idx, exp in EXPERIMENTS.items():
        if exp["category"] != current_category:
            current_category = exp["category"]
            print(f"\n  ── {current_category} {'─' * (50 - len(current_category))}")
        print(f"  [{idx}]  {exp['description']}")
        print(f"       Tempo stimato: {exp['estimated_time']}")

    print(f"\n  [0]  Esci")
    print("=" * 70)


def run_experiment(idx):
    """Lancia un esperimento specifico."""
    exp = EXPERIMENTS[idx]
    print(f"\n{'=' * 60}")
    print(f"  Lancio: {exp['name']}")
    print(f"  {exp['description']}")
    print(f"  Tempo stimato: {exp['estimated_time']}")
    print(f"{'=' * 60}\n")

    # Caso speciale: run_all_glue.py
    if exp["cmd"] == "__run_all_glue__":
        cmd = [sys.executable, "run_all_glue.py"]
    else:
        cmd = [sys.executable, "train.py"] + exp["cmd"]

    print(f"Comando: {' '.join(cmd)}\n")

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    status = "COMPLETATO" if result.returncode == 0 else f"FALLITO (codice {result.returncode})"
    print(f"\n{status} — {exp['name']} in {mins}m {secs}s")
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Riproduzione esperimenti Perceiver")
    parser.add_argument("--list", action="store_true", help="Mostra tutti gli esperimenti")
    parser.add_argument("--run", type=int, help="Lancia direttamente l'esperimento con questo numero")
    args = parser.parse_args()

    if args.list:
        print_menu()
        return

    if args.run:
        if args.run not in EXPERIMENTS:
            print(f"Errore: esperimento #{args.run} non trovato. Usa --list per vedere la lista.")
            sys.exit(1)
        run_experiment(args.run)
        return

    # Modo interattivo
    while True:
        print_menu()
        try:
            choice = input("\nSeleziona un esperimento (numero): ").strip()
            if not choice:
                continue
            choice = int(choice)
            if choice == 0:
                print("Uscita.")
                break
            if choice not in EXPERIMENTS:
                print(f"Scelta non valida: {choice}")
                continue

            confirm = input(f"\nConfermi il lancio di '{EXPERIMENTS[choice]['name']}'? (s/n): ").strip().lower()
            if confirm in ('s', 'si', 'sì', 'y', 'yes'):
                run_experiment(choice)
            else:
                print("Annullato.")
        except ValueError:
            print("Inserisci un numero valido.")
        except KeyboardInterrupt:
            print("\nInterrotto.")
            break


if __name__ == "__main__":
    main()
