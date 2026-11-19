# check.py
# Dashboard delle run v2: quali sono fatte, con quale metrica e split, quali sono
# divergite (nan) e quali mancano ancora. L'unica fonte e' logs/<id>/results.json:
# una run divergente ma protetta ha comunque un test_accuracy valido, preso dal
# checkpoint scelto prima della divergenza. Sola lettura, non lancia nulla.
#
#   python check.py            # tabella di stato di tutte le 42 run
#   python check.py --watch e01_baseline   # segue una run finche' non finisce o diverge
#   python check.py --csv results_reference.csv   # esporta i numeri di ogni results.json

import argparse
import json
import math
import os
import time

from experiments import EXPERIMENTS
from src.utils.run_metrics import reported_score

BY_ID = {e["id"]: e for e in EXPERIMENTS}


def _status_of(experiment_id):
    """Ritorna stato, punteggio con metrica e split, epoca."""
    path = os.path.join("logs", experiment_id, "results.json")
    if not os.path.exists(path):
        # C'e' un checkpoint ma niente json => run in corso o interrotta
        if os.path.isdir(os.path.join("logs", experiment_id)):
            return ("in corso/interrotta", None, None)
        return ("da fare", None, None)
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    score = reported_score(BY_ID.get(experiment_id, {}), data)
    ep = data.get("selected_epoch")
    if score.value is None or not math.isfinite(score.value):
        return ("DIVERGITA", score, ep)
    return ("ok", score, ep)


def dashboard():
    done = diverged = 0
    header = f"{'run':28s} {'gruppo':8s} {'stato':20s} {'epoca':>6s}  metrica / split: valore"
    print(header)
    print("-" * len(header))
    for eid, experiment in BY_ID.items():
        stato, score, ep = _status_of(eid)
        eps = str(ep) if ep is not None else "-"
        result = score.describe() if score else "-"
        print(f"{eid:28s} {experiment['group']:8s} {stato:20s} {eps:>6s}  {result}")
        if stato == "ok":
            done += 1
        elif stato == "DIVERGITA":
            diverged += 1
    print("-" * len(header))
    total = len(BY_ID)
    print(f"fatte: {done}/{total}   divergite: {diverged}   mancanti: {total - done - diverged}")
    if diverged:
        print("ATTENZIONE: run divergite presenti. Controlla grad_clip / lr prima di rifarle.")
    return diverged


def export_csv(path):
    """Scrive una riga per run con i campi di results.json: e' la tabella dei
    risultati di riferimento che accompagna il codice, senza dover rilanciare nulla."""
    import csv
    fields = ["run", "group", "status", "test_accuracy", "val_accuracy",
              "final_val_accuracy", "selected_epoch", "params", "seed"]
    n = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for eid, experiment in BY_ID.items():
            stato, _, _ = _status_of(eid)
            row = {"run": eid, "group": experiment["group"], "status": stato}
            json_path = os.path.join("logs", eid, "results.json")
            if os.path.exists(json_path):
                with open(json_path, encoding="utf-8") as handle_json:
                    data = json.load(handle_json)
                for key in fields[3:]:
                    value = data.get(key)
                    if isinstance(value, float) and key.endswith("accuracy"):
                        value = round(value, 4)
                    row[key] = value
                n += 1
            writer.writerow(row)
    print(f"{path}: {n} run con risultati su {len(BY_ID)}")


def watch(experiment_id, poll_seconds=30):
    """Segue una run: ogni poll_seconds ricontrolla results.json ed esce quando
    compare (fine) o quando riporta un'accuratezza nan (divergenza)."""
    print(f"watching {experiment_id} (Ctrl-C per uscire)")
    while True:
        stato, score, ep = _status_of(experiment_id)
        if stato == "ok":
            print(f"FINITA: {score.describe()} (checkpoint epoca {ep})")
            return 0
        if stato == "DIVERGITA":
            print(f"DIVERGITA: nessun valore finito per {score.metric} / {score.split}")
            return 1
        print(f"  [{stato}] ...", flush=True)
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Stato delle run Perceiver v2")
    parser.add_argument("--watch", type=str, help="segue una singola run finche' finisce/diverge")
    parser.add_argument("--poll", type=int, default=30, help="secondi fra un check e l'altro in --watch")
    parser.add_argument("--csv", type=str, metavar="FILE",
                        help="esporta i risultati di tutte le run in un CSV (una riga per run)")
    args = parser.parse_args()

    if args.csv:
        export_csv(args.csv)
        return
    if args.watch:
        raise SystemExit(watch(args.watch, args.poll))
    raise SystemExit(1 if dashboard() else 0)


if __name__ == "__main__":
    main()
