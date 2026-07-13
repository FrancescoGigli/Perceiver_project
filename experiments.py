# experiments.py
# Registro dichiarativo delle run di v2. Ogni voce e' un override della config base.
# Config base: M=1024, C=261, N=96, D=384, T=4, L=4, K=64, f_max=16, dropout 0, seed 42.

import argparse
import json
import os
import subprocess
import sys

BASE = [
    "--dataset", "cifar10",
    "--num_latents", "96",
    "--latent_dim", "384",
    "--num_cross_attend_stages", "4",
    "--num_transformer_blocks", "4",
    "--num_heads_cross", "1",
    "--num_heads_self", "8",
    "--fourier_num_bands", "64",
    "--fourier_max_freq", "16.0",
    "--latent_init_scale", "0.02",
    "--dropout", "0.0",
    "--optimizer", "lamb",
    "--lr", "0.004",
    "--scheduler", "multistep",
    "--epochs", "120",
    "--batch_size_cifar10", "64",
    "--patch_size", "1",
    "--val_split", "5000",
    "--no_amp",
]


def _exp(exp_id, group, overrides):
    return {"id": exp_id, "group": group, "overrides": overrides}


EXPERIMENTS = [
    # --- Tab. 1: il riferimento ---
    _exp("e01_baseline", "tab1", []),

    # --- Tab. 2: permutazione e tipo di positional encoding ---
    _exp("e02_permuted", "tab2", ["--permute_pixels", "--permute_pixels_seed", "42"]),
    _exp("e03_learned_pe", "tab2", ["--use_learned_pe", "--num_cross_attend_stages", "1"]),
    _exp("e04_learned_pe_permuted", "tab2",
         ["--use_learned_pe", "--num_cross_attend_stages", "1",
          "--permute_pixels", "--permute_pixels_seed", "42"]),

    # --- Tab. 5: senza latent transformer (cross-attend NON condivisi) ---
    _exp("e05_no_latent_T4", "tab5",
         ["--no_latent_transformer", "--no_share_cross_attend", "--num_cross_attend_stages", "4"]),
    _exp("e06_no_latent_T8", "tab5",
         ["--no_latent_transformer", "--no_share_cross_attend", "--num_cross_attend_stages", "8"]),
    _exp("e07_no_latent_T12", "tab5",
         ["--no_latent_transformer", "--no_share_cross_attend", "--num_cross_attend_stages", "12"]),

    # --- Tab. 6: numero di cross-attend x disposizione (T=4 interleaved = e01) ---
    _exp("e08_T1_interleaved", "tab6", ["--num_cross_attend_stages", "1"]),
    _exp("e09_T2_interleaved", "tab6", ["--num_cross_attend_stages", "2"]),
    _exp("e10_T8_interleaved", "tab6", ["--num_cross_attend_stages", "8"]),
    _exp("e11_T1_at_start", "tab6",
         ["--num_cross_attend_stages", "1", "--cross_attend_arrangement", "at_start"]),
    _exp("e12_T2_at_start", "tab6",
         ["--num_cross_attend_stages", "2", "--cross_attend_arrangement", "at_start"]),
    _exp("e13_T4_at_start", "tab6",
         ["--num_cross_attend_stages", "4", "--cross_attend_arrangement", "at_start"]),
    _exp("e14_T8_at_start", "tab6",
         ["--num_cross_attend_stages", "8", "--cross_attend_arrangement", "at_start"]),

    # --- Tab. 7: weight sharing (il ramo condiviso e' e01) ---
    _exp("e16_no_weight_sharing", "tab7", ["--no_weight_sharing"]),

    # --- Fig. 6: bande, frequenza massima, scala di inizializzazione ---
    _exp("e23_bands_4", "fig6", ["--fourier_num_bands", "4"]),
    _exp("e24_bands_16", "fig6", ["--fourier_num_bands", "16"]),
    _exp("e25_maxfreq_8", "fig6", ["--fourier_max_freq", "8.0"]),
    _exp("e26_maxfreq_64", "fig6", ["--fourier_max_freq", "64.0"]),   # 4x Nyquist, come v1
    _exp("e27_init_scale_0p1", "fig6", ["--latent_init_scale", "0.1"]),
    _exp("e28_init_scale_1p0", "fig6", ["--latent_init_scale", "1.0"]),

    # --- Fuori dal paper: la banda di rumore ---
    _exp("e31_baseline_seed1", "noise", ["--seed", "1"]),
    _exp("e32_baseline_seed2", "noise", ["--seed", "2"]),
]


def command_for(experiment):
    return [sys.executable, "train.py", *BASE, "--experiment_name", experiment["id"], *experiment["overrides"]]


def run(experiment_id):
    matches = [e for e in EXPERIMENTS if e["id"] == experiment_id]
    if not matches:
        raise SystemExit(f"esperimento sconosciuto: {experiment_id}")
    cmd = command_for(matches[0])
    print(" ".join(cmd))
    return subprocess.call(cmd)


def _run_status(experiment):
    """Ritorna (stato, test_acc, pulito) leggendo logs/<id>/results.json.
    'pulito' = risultato valido e non divergente (final_val > 0.5)."""
    path = os.path.join("logs", experiment["id"], "results.json")
    if not os.path.exists(path):
        return ("mancante", None, False)
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
    except (ValueError, OSError):
        return ("illeggibile", None, False)
    acc = data.get("test_accuracy")
    final_val = data.get("final_val_accuracy")
    if acc is None or acc != acc:  # None o nan
        return ("DIVERGITO (rifare)", acc, False)
    if final_val is not None and final_val <= 0.5:  # crollato al chance level
        return ("DIVERGITO (rifare)", acc, False)
    return ("OK", acc, True)


def next_experiment():
    """Stampa il riassunto di tutte le run e lancia la prima mancante/divergente."""
    print(f"{'#':<4}{'esperimento':<26}{'config':<42}{'stato':<20}{'test'}")
    print("-" * 100)
    first_todo = None
    n_ok = 0
    for i, exp in enumerate(EXPERIMENTS, 1):
        stato, acc, pulito = _run_status(exp)
        cfg = " ".join(exp["overrides"]) or "baseline (riferimento)"
        if len(cfg) > 40:
            cfg = cfg[:37] + "..."
        accs = f"{acc * 100:.2f}%" if isinstance(acc, (int, float)) and acc == acc else "-"
        print(f"{i:<4}{exp['id']:<26}{cfg:<42}{stato:<20}{accs:>7}")
        if pulito:
            n_ok += 1
        elif first_todo is None:
            first_todo = exp
    print("-" * 100)
    print(f"completi: {n_ok}/{len(EXPERIMENTS)}")
    if first_todo is None:
        print("Tutti gli esperimenti sono completi e puliti. Niente da lanciare.")
        return 0
    print(f"\n=> lancio il prossimo mancante: {first_todo['id']} "
          f"({' '.join(first_todo['overrides']) or 'baseline'})\n")
    return run(first_todo["id"])


def main():
    parser = argparse.ArgumentParser(description="Runner degli esperimenti Perceiver v2")
    parser.add_argument("--list", action="store_true", help="elenca gli esperimenti")
    parser.add_argument("--group", type=str, help="esegue tutti gli esperimenti di un gruppo")
    parser.add_argument("--run", type=str, help="esegue un singolo esperimento")
    parser.add_argument("--all", action="store_true", help="esegue tutte le 23 run in sequenza")
    parser.add_argument("--next", action="store_true",
                        help="mostra il riassunto e lancia il primo esperimento mancante/divergente")
    args = parser.parse_args()

    if args.list:
        for exp in EXPERIMENTS:
            print(f"{exp['id']:28s} {exp['group']:6s} {' '.join(exp['overrides'])}")
        return

    if args.next:
        raise SystemExit(next_experiment())

    if args.run:
        raise SystemExit(run(args.run))

    selected = [e for e in EXPERIMENTS if args.all or e["group"] == args.group]
    if not selected:
        raise SystemExit("niente da eseguire: usa --list, --run, --group o --all")

    for exp in selected:
        code = run(exp["id"])
        if code != 0:
            raise SystemExit(f"{exp['id']} e' terminato con codice {code}")


if __name__ == "__main__":
    main()
