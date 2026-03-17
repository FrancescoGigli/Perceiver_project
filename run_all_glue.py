# run_all_glue.py
# Launches all remaining GLUE sub-task experiments sequentially.
# Uses the same hyperparameters as the best SST-2 fine-tuning (v4).

import subprocess
import sys
import time

CHECKPOINT = "logs/mlm_wikitext103_optimized_batch32/checkpoints/last_checkpoint.pth"

# All GLUE tasks to run (excluding SST-2, already done)
TASKS = [
    {"dataset": "glue_cola",  "num_classes": 2, "name": "glue_cola_finetune"},
    {"dataset": "glue_mrpc",  "num_classes": 2, "name": "glue_mrpc_finetune"},
    {"dataset": "glue_stsb",  "num_classes": 1, "name": "glue_stsb_finetune"},
    {"dataset": "glue_qqp",   "num_classes": 2, "name": "glue_qqp_finetune"},
    {"dataset": "glue_mnli",  "num_classes": 3, "name": "glue_mnli_finetune"},
    {"dataset": "glue_qnli",  "num_classes": 2, "name": "glue_qnli_finetune"},
    {"dataset": "glue_rte",   "num_classes": 2, "name": "glue_rte_finetune"},
]

# Common hyperparameters (same as glue_sst2_finetune_v4)
COMMON_ARGS = [
    "--model_type", "perceiver_io",
    "--model_task", "classification",
    "--num_latents", "128",
    "--latent_dim", "512",
    "--num_cross_attend_stages", "1",
    "--num_transformer_blocks", "4",
    "--num_heads", "8",
    "--num_output_queries", "2",
    "--text_seq_len", "512",
    "--text_fourier_dim", "64",
    "--text_max_freq", "64.0",
    "--dropout", "0.1",
    "--optimizer", "lamb",
    "--lr", "0.0005",
    "--epochs", "30",
    "--batch_size_cifar10", "32",
    "--scheduler", "multistep",
    "--load_checkpoint_path", CHECKPOINT,
]

def run_task(task):
    name = task["name"]
    dataset = task["dataset"]
    
    print(f"\n{'='*60}")
    print(f"Starting: {name} ({dataset})")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, "train.py",
        "--dataset", dataset,
        "--experiment_name", name,
        *COMMON_ARGS,
    ]
    
    start_time = time.time()
    result = subprocess.run(cmd, cwd=".")
    elapsed = time.time() - start_time
    
    status = "SUCCESS" if result.returncode == 0 else f"FAILED (exit code {result.returncode})"
    print(f"\n{name}: {status} ({elapsed:.0f}s)")
    return result.returncode

if __name__ == "__main__":
    print("=" * 60)
    print("GLUE Benchmark - All Remaining Sub-tasks")
    print(f"Using checkpoint: {CHECKPOINT}")
    print(f"Tasks: {[t['name'] for t in TASKS]}")
    print("=" * 60)
    
    results = {}
    for task in TASKS:
        rc = run_task(task)
        results[task["name"]] = "OK" if rc == 0 else "FAIL"
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, status in results.items():
        print(f"  {name}: {status}")
