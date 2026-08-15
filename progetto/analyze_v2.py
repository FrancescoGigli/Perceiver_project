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

MODALITY_OF = {e["id"]: e.get("modality", "image") for e in EXPERIMENTS}
# Ogni modalita' si confronta col proprio baseline: ModelNet40 e' un altro task.
BASELINE_OF = {"image": "e01_baseline", "modelnet": "mn01_baseline"}

OUT_DIR = "analysis_results_v2"
OUT_CSV = os.path.join(OUT_DIR, "summary.csv")


def _is_missing(acc):
    """True se l'accuratezza e' None o nan (run divergita)."""
    return acc is None or (isinstance(acc, float) and math.isnan(acc))


def _acc(data):
    """Accuratezza da riportare: test_accuracy, o val_accuracy quando non c'e'
    una split di test separata (ModelNet40: la val E' l'insieme di test)."""
    acc = data.get("test_accuracy")
    if acc is None:
        acc = data.get("val_accuracy")
    return acc


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
        acc = _acc(data)
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
    # baseline per modalita': ogni run si confronta col riferimento del proprio task.
    baseline_acc = {}
    for mod, bid in BASELINE_OF.items():
        bd = results.get(bid)
        if bd is not None and not _is_missing(_acc(bd)):
            baseline_acc[mod] = _acc(bd)

    rows = []
    for eid, data in results.items():
        if eid not in GROUP_OF:
            # Cartella in logs/ che non corrisponde a nessuna run del registro
            # (prove, run di versioni precedenti): niente delta ne' verdetto, che
            # sarebbero senza senso rispetto a un baseline con cui non condivide
            # la configurazione. Elencata a parte in coda alla tabella.
            continue
        acc = _acc(data)
        epoch = data.get("selected_epoch")
        params = data.get("params")
        group = GROUP_OF.get(eid, "?")
        mod = MODALITY_OF.get(eid, "image")
        base = baseline_acc.get(mod)
        base_id = BASELINE_OF.get(mod)

        if _is_missing(acc):
            delta = None
            verdetto = "DIVERGITA"
        elif eid == base_id:
            delta = 0.0
            verdetto = "- (riferimento)"
        elif base is None:
            delta = None
            verdetto = "banda ignota"
        elif mod != "image":
            # la banda di rumore e' stimata solo sulle repliche immagine: per ModelNet
            # mostriamo il delta vs il suo baseline senza verdetto sulla banda.
            delta = acc - base
            verdetto = f"vs {base_id}"
        else:
            delta = acc - base
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


# Gli 8 task GLUE della Tab. 1 (WNLI escluso, come nel paper).
GLUE_TASKS_ORDER = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]


def _acc_of(results, run_id):
    data = results.get(run_id)
    if data is None:
        return None
    acc = _acc(data)
    return None if _is_missing(acc) else acc


def print_io_section(results, band):
    """Sintesi delle run Perceiver IO: confronto con il Perceiver su immagini,
    media GLUE (il numero confrontabile con la Tab. 1) e valore del pre-training."""
    print("\n\n=== PERCEIVER IO ===")

    # 1. Su classificazione il decoder a query dovrebbe pareggiare il pooling.
    # NB: architettura/dati identici a e01, ma ricetta di training del paper IO
    # (App. A.1: lr 1e-3 cosine): il multistep v1 con IO sotto-allena.
    perceiver = _acc_of(results, BASELINE_ID)
    print("\n-- Immagini: decoder a query (IO) vs pooling (Perceiver), stesso encoder --")
    if perceiver is None:
        print("   manca e01_baseline")
    else:
        print(f"   e01_baseline (Perceiver, pooling)      : {_fmt_pct(perceiver)}")
        for rid, label in (("io01_cifar", "io01_cifar (IO, ricetta paper)"),
                           ("io02_cifar_seed1", "io02_cifar_seed1 (replica, seed 1)")):
            acc = _acc_of(results, rid)
            if acc is None:
                print(f"   {label:<39}: run mancante")
                continue
            delta = acc - perceiver
            if band is None:
                verdetto = "banda ignota"
            else:
                verdetto = "sopra il rumore" if abs(delta) > band else "NON concludente"
            print(f"   {label:<39}: {_fmt_pct(acc)}   delta {_fmt_signed_pct(delta)}   {verdetto}")

    # 2. Media GLUE. Il paper riporta 81.0 (byte-level) / 81.1 (BERT su token).
    print("\n-- GLUE (paper Tab. 1: Perceiver IO byte-level 81.0, BERT 81.1) --")
    print(f"   {'task':8s} {'metrica':10s} {'da MLM':>9s} {'scratch':>9s} {'delta':>9s}")
    print("   " + "-" * 50)
    scores, missing = [], []
    for task in GLUE_TASKS_ORDER:
        pre = _acc_of(results, f"io_glue_{task}")
        scr = _acc_of(results, f"io_glue_{task}_scratch")
        if pre is None:
            missing.append(task)
        else:
            scores.append(pre)
        metrica = "Pearson" if task == "stsb" else "accuracy"
        delta = (pre - scr) if (pre is not None and scr is not None) else None
        print(f"   {task:8s} {metrica:10s} {_fmt_pct(pre):>9s} {_fmt_pct(scr):>9s} "
              f"{_fmt_signed_pct(delta):>9s}")
    print("   " + "-" * 50)
    if scores:
        media = sum(scores) / len(scores)
        print(f"   media su {len(scores)}/8 task: {media * 100:.2f}")
        if missing:
            print(f"   ATTENZIONE: media parziale, mancano {', '.join(missing)}")
        print("   NB: non e' esattamente il numero del paper: qui CoLA usa accuracy")
        print("       invece del coefficiente di Matthews. STS-B usa Pearson (come il paper).")
    else:
        print("   nessuna run GLUE completata")

    # 3. Il pre-training e' servito? Lo dicono i due controlli senza checkpoint.
    print("\n-- Quanto vale il pre-training MLM (run con e senza checkpoint) --")
    mlm = results.get("io_mlm")
    if mlm is not None and not _is_missing(_acc(mlm)):
        print(f"   io_mlm: accuratezza sui byte mascherati {_fmt_pct(_acc(mlm))} "
              f"(a caso: 0.39%, cioe' 1/256)")
    for task in ("sst2", "rte"):
        pre = _acc_of(results, f"io_glue_{task}")
        scr = _acc_of(results, f"io_glue_{task}_scratch")
        if pre is None or scr is None:
            print(f"   {task}: controllo incompleto")
        else:
            print(f"   {task}: {_fmt_pct(scr)} da zero -> {_fmt_pct(pre)} col pre-training "
                  f"({_fmt_signed_pct(pre - scr)})")


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

    extra = sorted(eid for eid in results if eid not in GROUP_OF)
    if extra:
        print(f"\nAltre cartelle in logs/ non presenti nel registro ({len(extra)}), "
              f"escluse dal confronto: {', '.join(extra)}")

    print_io_section(results, band)
    out_path = write_csv(rows)
    print(f"\nScritto: {out_path}")


if __name__ == "__main__":
    main()
