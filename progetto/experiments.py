# experiments.py
# Registro dichiarativo delle run di v2. Ogni voce e' un override della config base.
# Config base: M=1024, C=261, N=96, D=384, T=4, L=4, K=64, f_max=16, dropout 0, seed 42.

import argparse
import json
import os
import subprocess
import sys

# Config base per gli esperimenti su IMMAGINI (CIFAR-10).
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
]

# Config base per gli esperimenti su POINT CLOUD (ModelNet40, paper Tab. 4):
# 2 cross-attend, 6 self-attn per blocco, 2048 punti, LAMB lr 1e-3.
BASE_MODELNET = [
    "--dataset", "modelnet40",
    "--num_latents", "128",
    "--latent_dim", "512",
    "--num_cross_attend_stages", "2",
    "--num_transformer_blocks", "6",
    "--num_heads_cross", "1",
    "--num_heads_self", "8",
    "--dropout", "0.0",
    "--optimizer", "lamb",
    "--lr", "0.001",
    "--scheduler", "multistep",
    "--epochs", "120",
    "--batch_size_modelnet40", "32",
    "--modelnet40_num_points", "2048",
]


# --- Perceiver IO -----------------------------------------------------------
# Stesso encoder del Perceiver, ma l'output si legge con un array di QUERY che
# fa cross-attention sui latenti. Su classificazione la query e' una sola
# (equivale a un pooling appreso); nell'MLM le query sono una per posizione,
# ed e' li' che il decoder serve davvero (output strutturato).
# NB: il ramo perceiver_io usa --num_heads (uno solo), non lo split cross/self.

# Stessa architettura/dati di e01 (encoder identico, cambia solo il readout), ma
# con la RICETTA DI TRAINING del paper IO (App. A.1): lr basso con decadimento
# cosine. Il paper ha abbandonato esplicitamente il multistep del Perceiver v1
# per la classificazione IO; provato qui, il multistep a lr 4e-3 sotto-allena
# (val oscillante, train fermo) perche' il segnale attraverso la singola query
# di output e' piu' rumoroso del mean-pooling su 96 latenti.
BASE_IO_CIFAR = [
    "--dataset", "cifar10",
    "--model_type", "perceiver_io",
    "--num_output_queries", "1",
    "--num_latents", "96",
    "--latent_dim", "384",
    "--num_cross_attend_stages", "4",
    "--num_transformer_blocks", "4",
    "--num_heads", "8",
    "--fourier_num_bands", "64",
    "--fourier_max_freq", "16.0",
    "--latent_init_scale", "0.02",
    "--dropout", "0.0",
    "--optimizer", "lamb",
    "--lr", "0.001",
    "--scheduler", "cosine",
    "--epochs", "120",
    "--batch_size_cifar10", "64",
    "--patch_size", "1",
    "--val_split", "5000",
]

# Encoder condiviso da MLM e GLUE: la lunghezza di sequenza DEVE coincidere fra
# pre-training e fine-tuning, altrimenti i positional encoding non si trasferiscono.
_IO_TEXT_ENCODER = [
    "--model_type", "perceiver_io",
    "--num_latents", "128",
    "--latent_dim", "512",
    "--num_cross_attend_stages", "1",
    "--num_transformer_blocks", "4",
    "--num_heads", "8",
    "--dropout", "0.1",
    "--text_seq_len", "512",
    "--text_fourier_dim", "64",
    "--text_max_freq", "64.0",
    "--optimizer", "lamb",
    "--batch_size_cifar10", "32",   # riusato come batch dei dataset testuali
    "--num_workers", "2",           # la RAM di questa macchina non regge di piu'
]

# Pre-training MLM byte-level: niente tokenizer, vocabolario = 256 byte.
# num_output_queries lo imposta train.py = text_seq_len (una query per posizione).
BASE_IO_MLM = [
    *_IO_TEXT_ENCODER,
    "--dataset", "wikitext103",
    "--model_task", "mlm",
    "--mlm_vocab_size", "256",
    "--mlm_mask_prob", "0.15",
    "--lr", "0.001",
    # cosine e non multistep: le milestone di MultiStepLR sono fisse a [84,102,114]
    # (lo schedule del paper per le 120 epoche su CIFAR), quindi su una run corta
    # il learning rate non decadrebbe mai. Cosine usa davvero --epochs.
    "--scheduler", "cosine",
    "--epochs", "10",
]

# Fine-tuning GLUE: stessa architettura, testa di classificazione (1 query).
MLM_CHECKPOINT = "logs/io_mlm/checkpoints/best_model.pt"
BASE_IO_GLUE = [
    *_IO_TEXT_ENCODER,
    "--model_task", "classification",
    "--num_output_queries", "1",
    "--lr", "0.0005",
    "--scheduler", "cosine",   # come sopra: multistep non decadrebbe mai su run corte
    # --epochs non e' qui: lo imposta _glue() per taglia del task (vedi _GLUE_EPOCHS).
]


# Baseline non-Perceiver: il termine di paragone che il paper ha in Tab. 1 (ResNet-50,
# ViT) e che qui manca. Gira con uno script suo, non con train.py.
BASE_CNN = [
    "--epochs", "120",
    "--batch_size", "64",
    "--val_split", "5000",
]

# Multitask GLUE (paper IO Tab. 2): un solo modello, una query di output per task.
BASE_MULTITASK = [
    "--optimizer", "lamb",   # dichiarato qui come nelle altre basi, non nascosto nello script
    "--epochs", "10",
    "--batch_size", "32",
    "--lr", "0.0005",
    "--num_workers", "2",
    "--text_seq_len", "512",
    "--max_steps_per_epoch", "4000",   # senza, QQP e MNLI dominerebbero il campionamento
]


def _exp(exp_id, group, overrides, modality="image", script="train.py"):
    return {"id": exp_id, "group": group, "overrides": overrides,
            "modality": modality, "script": script}


# Epoche per taglia del task: i dataset GLUE vanno da 2.5k a 393k esempi. Tenerne 30
# ovunque significherebbe ~10 ore per MNLI e QQP (contro i 4 minuti di RTE) e per di
# piu' overfitting: il fine-tuning GLUE standard usa poche epoche sui dataset grandi.
_GLUE_EPOCHS = {
    "rte": 30, "mrpc": 30, "stsb": 30, "cola": 30,   # 2.5k - 8.5k esempi
    "sst2": 10, "qnli": 10,                           # 67k - 105k
    "qqp": 3, "mnli": 3,                              # 364k - 393k
}


def _glue(task, pretrained=True):
    """Run GLUE: dal checkpoint MLM (transfer) o da zero (controllo)."""
    suffix = "" if pretrained else "_scratch"
    overrides = ["--dataset", f"glue_{task}", "--epochs", str(_GLUE_EPOCHS[task])]
    if pretrained:
        overrides += ["--load_checkpoint_path", MLM_CHECKPOINT]
    return _exp(f"io_glue_{task}{suffix}", "io_glue", overrides, modality="io_glue")


EXPERIMENTS = [
    # --- Tab. 1: il riferimento ---
    _exp("e01_baseline", "tab1", []),

    # --- Tab. 2: permutazione e tipo di positional encoding ---
    _exp("e02_permuted", "tab2", ["--permute_pixels", "--permute_pixels_seed", "42"]),
    _exp("e03_learned_pe", "tab2", ["--use_learned_pe", "--num_cross_attend_stages", "1"]),
    _exp("e04_learned_pe_permuted", "tab2",
         ["--use_learned_pe", "--num_cross_attend_stages", "1",
          "--permute_pixels", "--permute_pixels_seed", "42"]),
    # Controllo che completa la Tab. 2: senza alcun positional encoding l'attenzione
    # e' invariante all'ordine, quindi il modello vede un insieme di pixel senza
    # struttura spaziale. E' il limite inferiore contro cui leggere gli altri tre.
    _exp("e29_no_pe", "tab2", ["--no_positional_encoding"]),

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

    # --- ModelNet40 (point cloud, paper Tab. 4): augmentation ---
    _exp("mn01_baseline", "modelnet", [], modality="modelnet"),
    _exp("mn02_rotation", "modelnet", ["--use_rotation"], modality="modelnet"),
    _exp("mn03_translation", "modelnet", ["--use_translation"], modality="modelnet"),

    # --- Perceiver IO su immagini: il decoder a query vs il pooling del Perceiver.
    # Stesso encoder di e01, ricetta di training del paper IO (vedi BASE_IO_CIFAR).
    # Il secondo seed stima la banda di rumore anche per IO.
    _exp("io01_cifar", "io_image", [], modality="io_image"),
    _exp("io02_cifar_seed1", "io_image", ["--seed", "1"], modality="io_image"),

    # --- Perceiver IO su linguaggio (paper Tab. 1): il caso in cui il decoder
    # serve davvero, un output per token mascherato, senza tokenizer.
    _exp("io_mlm", "io_mlm", [], modality="io_mlm"),

    # GLUE: transfer dal pre-training MLM, tutti e 8 i task (media confrontabile
    # con la Tab. 1 del paper).
    _glue("sst2"), _glue("cola"), _glue("mrpc"), _glue("stsb"),
    _glue("qqp"), _glue("mnli"), _glue("qnli"), _glue("rte"),

    # Controlli senza pre-training: misurano quanto vale l'MLM (uno grande, uno piccolo).
    _glue("sst2", pretrained=False),
    _glue("rte", pretrained=False),

    # --- Multitask GLUE (paper IO Tab. 2): gli 8 task insieme, una query per task.
    # E' il caso che mostra perche' il decoder a query e' diverso da un [CLS]:
    # aggiungere un task costa una query, non un token nell'input.
    _exp("io_glue_multitask", "io_glue", ["--load_checkpoint_path", MLM_CHECKPOINT],
         modality="io_multitask", script="multitask_glue.py"),

    # --- Baseline convoluzionale: senza un termine di paragone l'accuratezza del
    # Perceiver non e' leggibile (il paper la confronta con ResNet-50 e ViT).
    # Stesso split, stesso seed, stesse epoche delle run Perceiver.
    _exp("cnn_baseline", "baseline", [], modality="baseline", script="baseline_cnn.py"),
]


_BASE_FOR_MODALITY = {
    "image": BASE,
    "modelnet": BASE_MODELNET,
    "io_image": BASE_IO_CIFAR,
    "io_mlm": BASE_IO_MLM,
    "io_glue": BASE_IO_GLUE,   # il --dataset glue_<task> arriva dagli overrides
    "io_multitask": BASE_MULTITASK,
    "baseline": BASE_CNN,
}


def command_for(experiment):
    base = _BASE_FOR_MODALITY[experiment.get("modality", "image")]
    script = experiment.get("script", "train.py")
    return [sys.executable, script, *base, "--experiment_name", experiment["id"], *experiment["overrides"]]


def run(experiment_id):
    matches = [e for e in EXPERIMENTS if e["id"] == experiment_id]
    if not matches:
        raise SystemExit(f"esperimento sconosciuto: {experiment_id}")
    cmd = command_for(matches[0])
    print(" ".join(cmd))
    return subprocess.call(cmd)


# Soglia sotto la quale una run e' considerata crollata al livello del caso.
# Dipende dal task: 0.5 vale per CIFAR-10 (caso = 0.10) e ModelNet40 (0.025), non
# per GLUE, dove i task binari hanno il caso proprio a 0.50, MNLI a 0.33 e STS-B e'
# una regressione il cui "accuracy" e' -loss (negativo). Applicare 0.5 anche li'
# marcherebbe ogni run GLUE come divergita' e --next resterebbe in loop su di essa.
_COLLAPSE_FLOOR = {
    "image": 0.5,
    "io_image": 0.5,
    "modelnet": 0.5,
    "io_mlm": 0.05,   # MLM byte-level: il caso e' 1/256
    "io_glue": None,  # nessun controllo automatico sensato: si giudica a mano
}


def _run_status(experiment):
    """Ritorna (stato, test_acc, pulito) leggendo logs/<id>/results.json.
    'pulito' = risultato valido e non crollato al livello del caso per quel task."""
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
    # ModelNet40 non ha una split di test separata: la val E' l'insieme di test,
    # quindi test_accuracy resta null e il numero da riportare e' val_accuracy.
    if acc is None:
        acc = data.get("val_accuracy")
    if acc is None or acc != acc:  # None o nan: nessun risultato utilizzabile
        return ("DIVERGITO (rifare)", acc, False)
    # Se la validation finale e' crollata ma un risultato c'e', la run NON va rifatta:
    # il numero riportato viene dal checkpoint scelto sulla validation, che e'
    # metodologia corretta. Il crollo tardivo e' un dato da riportare (per
    # e28_init_scale_1p0 e' proprio il risultato della Fig. 6), non un errore; e
    # rilanciarla con lo stesso seed darebbe lo stesso esito, all'infinito.
    floor = _COLLAPSE_FLOOR.get(experiment.get("modality", "image"), 0.5)
    if floor is not None and final_val is not None and final_val <= floor:
        return ("OK (instabile a fine training)", acc, True)
    return ("OK", acc, True)


_MODALITY_TITLE = {
    "image": "=== PERCEIVER — IMMAGINI (CIFAR-10) ===",
    "modelnet": "=== PERCEIVER — POINT CLOUD (ModelNet40) ===",
    "io_image": "=== PERCEIVER IO — IMMAGINI (CIFAR-10) ===",
    "io_mlm": "=== PERCEIVER IO — PRE-TRAINING MLM byte-level (WikiText-103) ===",
    "io_glue": "=== PERCEIVER IO — FINE-TUNING GLUE ===",
}


def next_experiment():
    """Stampa il riassunto (diviso per modalita') e lancia la prima mancante/divergente."""
    first_todo = None
    n_ok = 0
    last_modality = None
    for i, exp in enumerate(EXPERIMENTS, 1):
        modality = exp.get("modality", "image")
        if modality != last_modality:
            print(f"\n{_MODALITY_TITLE.get(modality, modality)}")
            print(f"{'#':<4}{'esperimento':<26}{'config':<42}{'stato':<20}{'risultato'}")
            print("-" * 100)
            last_modality = modality
        stato, acc, pulito = _run_status(exp)
        cfg = " ".join(exp["overrides"]) or "baseline (riferimento)"
        if len(cfg) > 40:
            cfg = cfg[:37] + "..."
        accs = f"{acc * 100:.2f}%" if isinstance(acc, (int, float)) and acc == acc else "-"
        print(f"{i:<4}{exp['id']:<26}{cfg:<42}{stato:<20}{accs:>9}")
        if pulito:
            n_ok += 1
        elif first_todo is None:
            first_todo = exp
    total = len(EXPERIMENTS)
    counts = {}
    for e in EXPERIMENTS:
        m = e.get("modality", "image")
        counts[m] = counts.get(m, 0) + 1
    per_modality = "   ".join(f"{m}: {n}" for m, n in counts.items())
    print("-" * 100)
    print(f"completi: {n_ok}/{total}   ({per_modality})")
    if first_todo is None:
        print("Tutti gli esperimenti sono completi e puliti. Niente da lanciare.")
        return 0
    print(f"\n=> lancio il prossimo mancante: {first_todo['id']} "
          f"[{first_todo.get('modality', 'image')}] "
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
