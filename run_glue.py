# run_glue.py
# Launcher for individual GLUE experiments.
# Usage:
#   python run_glue.py cola           → fine-tune CoLA
#   python run_glue.py sst2           → fine-tune SST-2
#   python run_glue.py mnli --epochs 15 --lr 0.001
#   python run_glue.py --list         → show all available tasks

import subprocess
import sys
import argparse
import time

CHECKPOINT = "logs/mlm_wikitext103_optimized_batch32/checkpoints/last_checkpoint.pth"

AVAILABLE_TASKS = {
    "cola":  {"dataset": "glue_cola",  "classes": 2, "type": "classification"},
    "mrpc":  {"dataset": "glue_mrpc",  "classes": 2, "type": "classification"},
    "stsb":  {"dataset": "glue_stsb",  "classes": 1, "type": "regression"},
    "qqp":   {"dataset": "glue_qqp",   "classes": 2, "type": "classification"},
    "mnli":  {"dataset": "glue_mnli",  "classes": 3, "type": "classification"},
    "qnli":  {"dataset": "glue_qnli",  "classes": 2, "type": "classification"},
    "rte":   {"dataset": "glue_rte",   "classes": 2, "type": "classification"},
    "sst2":  {"dataset": "glue_sst2",  "classes": 2, "type": "classification"},
}

def main():
    parser = argparse.ArgumentParser(description="Launch individual GLUE fine-tuning experiments")
    parser.add_argument("task", nargs="?", type=str, help="GLUE task name (cola, mrpc, stsb, qqp, mnli, qnli, rte, sst2)")
    parser.add_argument("--list", action="store_true", help="List all available tasks")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate (default: 0.0005)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT, help="Path to MLM checkpoint")
    parser.add_argument("--suffix", type=str, default="", help="Suffix for experiment name (e.g. _v2)")
    parser.add_argument("--no_checkpoint", action="store_true", help="Train from scratch (no pretrained weights)")
    args = parser.parse_args()

    if args.list or not args.task:
        print("\nAvailable GLUE tasks:")
        print("-" * 50)
        for name, info in AVAILABLE_TASKS.items():
            print(f"  {name:6s}  →  {info['classes']} classes  ({info['type']})")
        print(f"\nUsage:  python run_glue.py <task> [--epochs N] [--lr LR]")
        print(f"Example: python run_glue.py cola --epochs 15")
        return

    task_name = args.task.lower()
    if task_name not in AVAILABLE_TASKS:
        print(f"Error: unknown task '{task_name}'. Use --list to see available tasks.")
        sys.exit(1)

    task_info = AVAILABLE_TASKS[task_name]
    exp_name = f"glue_{task_name}_finetune{args.suffix}"

    print(f"\n{'='*60}")
    print(f"  GLUE Fine-tuning: {task_name.upper()}")
    print(f"  Experiment: {exp_name}")
    print(f"  Classes: {task_info['classes']}  Type: {task_info['type']}")
    print(f"  Epochs: {args.epochs}  LR: {args.lr}  Batch: {args.batch_size}")
    if not args.no_checkpoint:
        print(f"  Checkpoint: {args.checkpoint}")
    else:
        print(f"  Checkpoint: None (training from scratch)")
    print(f"{'='*60}\n")

    cmd = [
        sys.executable, "train.py",
        "--dataset", task_info["dataset"],
        "--experiment_name", exp_name,
        "--model_type", "perceiver_io",
        "--model_task", "classification",
        "--num_latents", "128",
        "--latent_dim", "512",
        "--num_cross_attend_stages", "1",
        "--num_transformer_blocks", "4",
        "--num_heads", "8",
        "--num_output_queries", "2",
        "--text_seq_len", str(args.seq_len),
        "--text_fourier_dim", "64",
        "--text_max_freq", "64.0",
        "--dropout", "0.1",
        "--optimizer", "lamb",
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--batch_size_cifar10", str(args.batch_size),
        "--scheduler", "multistep",
    ]

    if not args.no_checkpoint:
        cmd.extend(["--load_checkpoint_path", args.checkpoint])

    start = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - start

    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    status = "✅ SUCCESS" if result.returncode == 0 else f"❌ FAILED (code {result.returncode})"
    print(f"\n{status} — {task_name.upper()} completed in {mins}m {secs}s")

if __name__ == "__main__":
    main()
