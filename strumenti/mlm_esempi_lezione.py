# strumenti/mlm_esempi_lezione.py
# Esempi per la lezione interattiva (cap. 46): brani veri del file su cui valida
# io_mlm (WikiText-103, test.csv), con il 15% dei byte mascherato, e le risposte
# di tre regole: il modello io_mlm, la tabella dei vicini di baseline_mlm.py e
# «sempre il byte più frequente» (lo spazio).
#
#   python strumenti/mlm_esempi_lezione.py
#
# Serve logs/io_mlm/checkpoints/best_model.pt e il dataset in progetto/data.
# Gira su CPU in circa un minuto e scrive sito/lezione/js/mlm-esempi.js.

import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

QUI = os.path.dirname(os.path.abspath(__file__))
PROGETTO = os.path.join(QUI, "..", "progetto")
USCITA = os.path.join(QUI, "..", "sito", "lezione", "js", "mlm-esempi.js")
sys.path.insert(0, PROGETTO)
os.chdir(PROGETTO)

from baseline_mlm import count_triples, data_paths, load_bytes  # noqa: E402
from src.perceiver_io.perceiver_io import PerceiverIO  # noqa: E402
from src.utils.positional_encoding import FourierPositionalEncoding  # noqa: E402

SEQ, MASK_P, N_ESEMPI = 512, 0.15, 5

train_path, valid_path = data_paths("./data")
valid = load_bytes(valid_path)
triples = count_triples(load_bytes(train_path))
piu_frequente = int(triples.sum(axis=(0, 2)).argmax())
dato_sinistro = triples.sum(axis=2).argmax(axis=1)
dato_destro = triples.sum(axis=0).argmax(axis=0)
dati_entrambi = triples.argmax(axis=1)
visti_entrambi = triples.max(axis=1) > 0


def vicini(x, masked):
    """La regola di baseline_mlm.evaluate, applicata a una finestra."""
    pred = {}
    for i in np.flatnonzero(masked):
        sx = i > 0 and not masked[i - 1]
        dx = i < SEQ - 1 and not masked[i + 1]
        if sx and dx:
            coppia = (x[i - 1], x[i + 1])
            pred[i] = dati_entrambi[coppia] if visti_entrambi[coppia] else dato_sinistro[x[i - 1]]
        elif sx:
            pred[i] = dato_sinistro[x[i - 1]]
        elif dx:
            pred[i] = dato_destro[x[i + 1]]
        else:
            pred[i] = piu_frequente
    return {int(i): int(p) for i, p in pred.items()}


# Il modello come lo costruisce train.py per io_mlm (logs/io_mlm/config.txt).
model = PerceiverIO(input_dim=270, num_classes=256, num_latents=128, latent_dim=512,
                    num_cross_attend_stages=1, num_transformer_blocks=4, num_heads=8,
                    head_dim=64, mlp_ratio=4, dropout=0.1, num_output_queries=SEQ,
                    task="mlm", mlm_vocab_size=256, save_attention_maps=False,
                    weight_sharing=True, latent_init_scale=0.02, input_pe=None)
ckpt = torch.load(os.path.join("logs", "io_mlm", "checkpoints", "best_model.pt"),
                  map_location="cpu", weights_only=True)
model.load_state_dict(ckpt.get("model_state_dict", ckpt))
model.eval()

# Come WikiText103PerceiverDataModule: one-hot da 257 (256 byte + [MASK]) e Fourier 1D a 6 bande.
pe = FourierPositionalEncoding(num_bands=6, max_freq=64.0, num_pos_feats=1)
with torch.no_grad():
    posizioni = pe((torch.arange(SEQ).float() / (SEQ - 1)).unsqueeze(-1))


def modello(x, masked):
    ids = torch.tensor(x, dtype=torch.long)
    ids[torch.tensor(masked)] = 256
    inp = torch.cat([F.one_hot(ids, 257).float(), posizioni], dim=-1).unsqueeze(0)
    with torch.no_grad():
        prob = model(inp).reshape(SEQ, -1).softmax(-1)
    top = prob.argmax(-1)
    return {int(i): (int(top[i]), round(float(prob[i, top[i]]), 3)) for i in np.flatnonzero(masked)}


# Finestre allineate come nel data module; si tengono quelle di sola prosa ASCII.
buone = []
for w in range(len(valid) // SEQ):
    x = valid[w * SEQ:(w + 1) * SEQ]
    if x.max() >= 127 or x.min() < 32:
        continue
    testo = bytes(x).decode("ascii")
    if "@" in testo or " = " in testo or '"' in testo:
        continue
    if sum(c.isalpha() for c in testo) / SEQ > 0.78:
        buone.append(w)

scelte = sorted(np.random.default_rng(2026).choice(buone, size=N_ESEMPI, replace=False).tolist())
esempi = []
for k, w in enumerate(scelte):
    x = valid[w * SEQ:(w + 1) * SEQ].astype(np.int64)
    masked = np.random.default_rng(100 + k).random(SEQ) < MASK_P
    pm, pv = modello(x, masked), vicini(x, masked)
    # [posizione, byte vero, byte del modello, sua probabilita', byte della tabella dei vicini]
    maschere = [[int(i), int(x[i]), pm[i][0], pm[i][1], pv[i]] for i in np.flatnonzero(masked)]
    esempi.append({"finestra": int(w), "testo": bytes(x.astype(np.uint8)).decode("ascii"),
                   "maschere": maschere})
    giusti = [sum(m[1] == m[j] for m in maschere) for j in (2, 4)]
    print(f"finestra {w}: {len(maschere)} byte mascherati, modello {giusti[0]}, vicini {giusti[1]}")

dati = {"fonte": "WikiText-103 (test.csv, la validation di io_mlm)", "seq_len": SEQ,
        "mask_prob": MASK_P, "byte_piu_frequente": piu_frequente, "esempi": esempi}
with open(USCITA, "w", encoding="utf-8", newline="\n") as fh:
    fh.write("// Generato da strumenti/mlm_esempi_lezione.py: non modificare a mano.\n")
    fh.write("window.MLM_ESEMPI = " + json.dumps(dati, ensure_ascii=True) + ";\n")
print("scritto", os.path.normpath(USCITA))
