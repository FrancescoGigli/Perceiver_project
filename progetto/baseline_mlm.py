# baseline_mlm.py
# Baseline senza rete per il pre-training MLM byte-level (run io_mlm).
#
#   python baseline_mlm.py
#
# Lo 0,39% (1/256) e' il caso uniforme, ma il testo non e' uniforme: lo spazio da
# solo e' circa un byte su cinque. Qui si misura quanto si indovina di un byte
# mascherato con due regole che non imparano nulla per gradiente:
#   1. rispondere sempre il byte piu' frequente del training;
#   2. una tabella dei vicini: il byte piu' frequente fra quello prima e quello
#      dopo, contato sul training. Se un vicino e' mascherato si usa l'altro,
#      se lo sono entrambi si torna alla regola 1.
# Stessi file e stessa lettura di src/data/wikitext103.py, stesse finestre da 512
# byte, 15% dei byte mascherati in modo indipendente. Sola lettura, niente GPU.

import argparse
import io
import json
import os

import numpy as np

VOCAB = 256


def data_paths(data_dir):
    """Gli stessi file che legge WikiText103PerceiverDataModule.setup()."""
    base = os.path.join(data_dir, "wikitext-103", "wikitext-103")
    train = os.path.join(base, "wiki.train.tokens")
    valid = os.path.join(base, "wiki.valid.tokens")
    if not os.path.exists(train) and os.path.exists(os.path.join(base, "train.csv")):
        # Versione fast.ai del dataset: non ha un file di validation, il data
        # module valida su test.csv. Lo stesso vale qui.
        train = os.path.join(base, "train.csv")
        valid = os.path.join(base, "test.csv")
    return train, valid


def load_bytes(path):
    """Come WikiText103PerceiverDataModule._load_bytes: testo utf-8 riportato a byte."""
    with io.open(path, "r", encoding="utf-8") as handle:
        return np.frombuffer(handle.read().encode("utf-8"), dtype=np.uint8)


def count_triples(data, chunk=1 << 25):
    """Conteggi delle terne (prima, centro, dopo): tabella [256, 256, 256]."""
    counts = np.zeros(VOCAB ** 3, dtype=np.int64)
    n = len(data)
    for start in range(0, n - 2, chunk):
        stop = min(start + chunk, n - 2)
        w = data[start:stop + 2].astype(np.int64)
        counts += np.bincount((w[:-2] * VOCAB + w[1:-1]) * VOCAB + w[2:],
                              minlength=VOCAB ** 3)
    return counts.reshape(VOCAB, VOCAB, VOCAB)


def evaluate(valid, triples, seq_len, mask_prob, seed):
    """Accuratezza sui byte mascherati di ogni regola, sulle finestre del data module."""
    n = (len(valid) // seq_len) * seq_len          # la coda che non riempie una finestra resta fuori
    x = valid[:n].astype(np.int64)
    masked = np.random.default_rng(seed).random(n) < mask_prob
    pos = np.flatnonzero(masked)
    offset = pos % seq_len
    target = x[pos]

    # Un vicino conta solo se sta nella stessa finestra e non e' mascherato anche lui.
    has_left = offset > 0
    has_left[has_left] = ~masked[pos[has_left] - 1]
    has_right = offset < seq_len - 1
    has_right[has_right] = ~masked[pos[has_right] + 1]
    left = x[np.maximum(pos - 1, 0)]
    right = x[np.minimum(pos + 1, n - 1)]

    most_frequent = triples.sum(axis=(0, 2)).argmax()
    given_left = triples.sum(axis=2).argmax(axis=1)       # [prima] -> centro
    given_right = triples.sum(axis=0).argmax(axis=0)      # [dopo]  -> centro
    given_both = triples.argmax(axis=1)                   # [prima, dopo] -> centro
    seen_both = triples.max(axis=1) > 0

    pred = np.full(len(pos), most_frequent)
    only_left = has_left & ~has_right
    only_right = has_right & ~has_left
    both = has_left & has_right
    pred[only_left] = given_left[left[only_left]]
    pred[only_right] = given_right[right[only_right]]
    seen = both & seen_both[left, right]
    pred[seen] = given_both[left[seen], right[seen]]
    unseen = both & ~seen_both[left, right]
    pred[unseen] = given_left[left[unseen]]

    return {
        "windows": n // seq_len,
        "masked": len(pos),
        "most_frequent_byte": int(most_frequent),
        "uniform": 1.0 / VOCAB,
        "always_most_frequent": float((target == most_frequent).mean()),
        "neighbours": float((pred == target).mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Baseline senza rete per l'MLM byte-level")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--text_seq_len", type=int, default=512)
    parser.add_argument("--mlm_mask_prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path, valid_path = data_paths(args.data_dir)
    for path in (train_path, valid_path):
        if not os.path.exists(path):
            raise SystemExit(f"File non trovato: {path}\n"
                             f"Scarica prima il dataset con: python experiments.py --run io_mlm")

    print(f"conteggi sul training: {train_path}")
    triples = count_triples(load_bytes(train_path))
    res = evaluate(load_bytes(valid_path), triples,
                   args.text_seq_len, args.mlm_mask_prob, args.seed)

    print(f"misura su: {valid_path}  ({res['windows']:,} finestre da {args.text_seq_len} byte, "
          f"{res['masked']:,} byte mascherati)\n")
    print(f"  caso uniforme (1/256)                     {res['uniform'] * 100:6.2f}%")
    print(f"  sempre il byte piu' frequente ({chr(res['most_frequent_byte'])!r})       "
          f"{res['always_most_frequent'] * 100:6.2f}%")
    print(f"  tabella dei vicini (byte prima e dopo)    {res['neighbours'] * 100:6.2f}%")

    run = os.path.join("logs", "io_mlm", "results.json")
    if os.path.exists(run):
        with open(run, encoding="utf-8") as handle:
            acc = json.load(handle)["val_accuracy"]
        print(f"  io_mlm, Perceiver IO (results.json)       {acc * 100:6.2f}%")


if __name__ == "__main__":
    main()
