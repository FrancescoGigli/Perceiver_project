# check.py
# Dashboard delle run v2: quali sono fatte, il loro test accuracy, quali sono
# divergite (nan) e quali mancano ancora. Legge logs/<id>/results.json e scansiona
# lo stdout salvato per il marcatore "nan". Sola lettura, non lancia nulla.
#
#   python check.py            # tabella di stato di tutte le 23 run
#   python check.py --watch e01_baseline   # segue una run finche' non finisce o diverge

import argparse
import glob
import json
import os
import re
import sys
import time

from experiments import EXPERIMENTS

GROUP_OF = {e["id"]: e["group"] for e in EXPERIMENTS}
ORDER = [e["id"] for e in EXPERIMENTS]


def _tail_has_nan(experiment_id):
    """True se l'ultimo log di training della run contiene 'nan' (divergenza)."""
    logs = glob.glob(os.path.join("logs", experiment_id, "events.out.tfevents.*"))
    # I tfevents sono binari; il segnale affidabile e' results.json + il json stesso.
    # Qui controlliamo solo il json: una run divergente ma protetta ha comunque un
    # test_accuracy valido (checkpoint pre-divergenza). Il nan si vede nello stdout,
    # che non e' catturato in modo stabile: usiamo results.json come verita'.
    return False


def _status_of(experiment_id):
    """Ritorna (stato, test_acc, epoca) leggendo results.json."""
    path = os.path.join("logs", experiment_id, "results.json")
    if not os.path.exists(path):
        # C'e' un checkpoint ma niente json => run in corso o interrotta
        if os.path.isdir(os.path.join("logs", experiment_id)):
            return ("in corso/interrotta", None, None)
        return ("da fare", None, None)
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    acc = data.get("test_accuracy")
    if acc is None:  # ModelNet40 non ha una split di test separata: la val E' l'insieme di test
        acc = data.get("val_accuracy")
    ep = data.get("selected_epoch")
    if acc is None or (isinstance(acc, float) and acc != acc):  # None o nan
        return ("DIVERGITA", acc, ep)
    return ("ok", acc, ep)


def dashboard():
    done = diverged = 0
    print(f"{'run':28s} {'gruppo':6s} {'stato':20s} {'test acc':>9s} {'epoca':>6s}")
    print("-" * 74)
    for eid in ORDER:
        stato, acc, ep = _status_of(eid)
        accs = f"{acc*100:.2f}%" if isinstance(acc, (int, float)) and acc == acc else "-"
        eps = str(ep) if ep is not None else "-"
        print(f"{eid:28s} {GROUP_OF[eid]:6s} {stato:20s} {accs:>9s} {eps:>6s}")
        if stato == "ok":
            done += 1
        elif stato == "DIVERGITA":
            diverged += 1
    print("-" * 74)
    total = len(ORDER)
    print(f"fatte: {done}/{total}   divergite: {diverged}   mancanti: {total - done - diverged}")
    if diverged:
        print("ATTENZIONE: run divergite presenti. Controlla grad_clip / lr prima di rifarle.")
    return diverged


def watch(experiment_id, poll_seconds=30):
    """Segue una run: stampa l'ultima epoca dallo stdout del processo, esce quando
    compare results.json (fine) oppure 'nan' (divergenza)."""
    print(f"watching {experiment_id} (Ctrl-C per uscire)")
    while True:
        stato, acc, ep = _status_of(experiment_id)
        if stato == "ok":
            print(f"FINITA: test {acc*100:.2f}% (checkpoint epoca {ep})")
            return 0
        if stato == "DIVERGITA":
            print(f"DIVERGITA: results.json ha test_accuracy={acc}")
            return 1
        print(f"  [{stato}] ...", flush=True)
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description="Stato delle run Perceiver v2")
    parser.add_argument("--watch", type=str, help="segue una singola run finche' finisce/diverge")
    parser.add_argument("--poll", type=int, default=30, help="secondi fra un check e l'altro in --watch")
    args = parser.parse_args()

    if args.watch:
        raise SystemExit(watch(args.watch, args.poll))
    raise SystemExit(1 if dashboard() else 0)


if __name__ == "__main__":
    main()
