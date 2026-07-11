# analyze_v2.py
# Analisi dei risultati finali delle run v2: tabella comparativa, banda di
# rumore, verdetto per ogni run rispetto al baseline. Sola lettura, non lancia
# nulla (vedi anche check.py, che e' invece un dashboard di STATO delle run).
#
#   python analyze_v2.py
#
# Legge logs/<id>/results.json, incrocia col registro in experiments.py
# (id -> gruppo). La banda di rumore e' l'escursione (max-min) del
# test_accuracy fra le repliche note del baseline (e01_baseline,
# e31_baseline_seed1, e32_baseline_seed2): quelle che esistono fra queste tre.
# Scrive analysis_results_v2/summary.csv.

import csv
import glob
import json
import math
import os

from experiments import EXPERIMENTS

GROUP_OF = {e["id"]: e["group"] for e in EXPERIMENTS}
ORDER = {e["id"]: i for i, e in enumerate(EXPERIMENTS)}

NOISE_BAND_IDS = ["e01_baseline", "e31_baseline_seed1", "e32_baseline_seed2"]
BASELINE_ID = "e01_baseline"

OUT_DIR = "analysis_results_v2"
OUT_CSV = os.path.join(OUT_DIR, "summary.csv")


def _is_missing(acc):
    """True se test_accuracy e' None o nan (run divergita)."""
    return acc is None or (isinstance(acc, float) and math.isnan(acc))


def load_results(log_dir="logs"):
    """Legge tutti i logs/*/results.json trovati. Ritorna {id: dict}."""
    results = {}
    for path in sorted(glob.glob(os.path.join(log_dir, "*", "results.json"))):
        exp_id = os.path.basename(os.path.dirname(path))
        with open(path, encoding="utf-8") as handle:
            results[exp_id] = json.load(handle)
    return results


def noise_band(results):
    """Escursione (max-min) del test_accuracy fra le repliche note del baseline
    che esistono davvero. Ritorna (banda_o_None, lista_id_usati)."""
    used = []
    accs = []
    for eid in NOISE_BAND_IDS:
        data = results.get(eid)
        if data is None:
            continue
        acc = data.get("test_accuracy")
        if _is_missing(acc):
            continue
        used.append(eid)
        accs.append(acc)
    if len(accs) < 2:
        return None, used
    return max(accs) - min(accs), used


def build_rows(results, band):
    """Per ogni run con results.json: id, gruppo, test_acc, epoca, params,
    delta rispetto a e01_baseline, verdetto sulla banda di rumore."""
    baseline_data = results.get(BASELINE_ID)
    baseline_acc = None
    if baseline_data is not None and not _is_missing(baseline_data.get("test_accuracy")):
        baseline_acc = baseline_data["test_accuracy"]

    rows = []
    for eid, data in results.items():
        acc = data.get("test_accuracy")
        epoch = data.get("selected_epoch")
        params = data.get("params")
        group = GROUP_OF.get(eid, "?")

        if _is_missing(acc):
            delta = None
            verdetto = "DIVERGITA"
        elif eid == BASELINE_ID:
            delta = 0.0
            verdetto = "-"  # e' il riferimento, il confronto con se stesso non ha senso
        elif baseline_acc is None:
            delta = None
            verdetto = "banda ignota"
        else:
            delta = acc - baseline_acc
            if band is None:
                verdetto = "banda ignota"
            elif abs(delta) > band:
                verdetto = "sopra il rumore"
            else:
                verdetto = "NON concludente"

        rows.append(
            {
                "id": eid,
                "group": group,
                "test_acc": acc,
                "selected_epoch": epoch,
                "params": params,
                "delta_vs_e01": delta,
                "verdetto": verdetto,
            }
        )

    rows.sort(key=lambda r: ORDER.get(r["id"], len(ORDER)))
    return rows


def write_csv(rows, path=OUT_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "group", "test_acc", "selected_epoch", "params", "delta_vs_e01", "verdetto"])
        for r in rows:
            writer.writerow(
                [
                    r["id"],
                    r["group"],
                    "" if _is_missing(r["test_acc"]) else r["test_acc"],
                    "" if r["selected_epoch"] is None else r["selected_epoch"],
                    "" if r["params"] is None else r["params"],
                    "" if r["delta_vs_e01"] is None else r["delta_vs_e01"],
                    r["verdetto"],
                ]
            )
    return path


def _fmt_pct(value):
    if isinstance(value, (int, float)) and value == value:  # non-nan
        return f"{value * 100:.2f}%"
    return "-"


def _fmt_signed_pct(value):
    if isinstance(value, (int, float)) and value == value:
        return f"{value * 100:+.2f}%"
    return "-"


def print_table(rows, band, band_ids):
    header = f"{'id':28s} {'gruppo':7s} {'test acc':>9s} {'epoca':>6s} {'params':>11s} {'delta':>8s}  {'verdetto':16s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        params = r["params"]
        params_s = f"{params:,}" if isinstance(params, (int, float)) else "-"
        epoch_s = str(r["selected_epoch"]) if r["selected_epoch"] is not None else "-"
        print(
            f"{r['id']:28s} {r['group']:7s} {_fmt_pct(r['test_acc']):>9s} {epoch_s:>6s} "
            f"{params_s:>11s} {_fmt_signed_pct(r['delta_vs_e01']):>8s}  {r['verdetto']:16s}"
        )
    print("-" * len(header))
    if band is None:
        print(
            f"Banda di rumore: IGNOTA (repliche disponibili: {band_ids or 'nessuna'}, "
            f"servono almeno 2 fra {NOISE_BAND_IDS})"
        )
    else:
        print(f"Banda di rumore (escursione max-min fra {band_ids}): {band * 100:.2f} punti percentuali")


def main():
    results = load_results()
    if not results:
        print(
            "Nessun results.json trovato sotto logs/*/results.json. "
            "Nessuna run e' ancora completata: niente da analizzare."
        )
        return

    band, band_ids = noise_band(results)
    rows = build_rows(results, band)

    print_table(rows, band, band_ids)
    out_path = write_csv(rows)
    print(f"\nScritto: {out_path}")


if __name__ == "__main__":
    main()
