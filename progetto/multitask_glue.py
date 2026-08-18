# multitask_glue.py
# Perceiver IO multitask su GLUE (paper Tab. 2): un solo modello addestrato in
# contemporanea su tutti gli 8 task, con UNA QUERY DI OUTPUT PER TASK.
#
#   python multitask_glue.py --experiment_name io_glue_multitask \
#       --load_checkpoint_path logs/io_mlm/checkpoints/best_model.pt
#
# E' il caso che mostra la proprieta' distintiva del decoder di Perceiver IO:
# l'array di output e' indipendente dall'input, quindi per aggiungere un task basta
# una query in piu' -- non serve infilare un token [CLS] nella sequenza di ingresso.
# Nel paper il multitask (81.8) supera il single-task (81.0).
#
# Scrive logs/<id>/results.json nello stesso formato delle altre run: val_accuracy
# e' la media sugli 8 task, e per_task riporta il dettaglio.

import argparse
import json
import os

import numpy as np
import torch
import torch_optimizer as custom_optim  # LAMB, come train.py
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset
from tqdm import tqdm

from src.data.glue_sst2 import SST2PerceiverDataModule
from src.data.glue_tasks import GLUEPerceiverDataModule
from src.perceiver_io.perceiver_io import PerceiverIO
from src.utils.seed import set_global_seed

# Ordine fisso: l'indice nella lista E' l'indice della query di output del task.
TASKS = ["cola", "sst2", "mrpc", "stsb", "qqp", "mnli", "qnli", "rte"]
STSB = TASKS.index("stsb")
VOCAB = 257  # 256 byte + token di mask, come nel pre-training MLM


class _Tagged(Dataset):
    """Dataset di un task, con l'indice del task attaccato a ogni esempio."""

    def __init__(self, base, task_idx):
        self.base = base
        self.task_idx = task_idx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        # label sempre float: in un batch misto convivono task di classificazione
        # (interi) e la regressione di STS-B. Il cast avviene nella loss.
        return x, torch.tensor(float(y)), self.task_idx


def build_datamodules(args):
    dms = {}
    for task in TASKS:
        common = dict(data_dir=args.data_dir, batch_size=args.batch_size,
                      num_workers=args.num_workers, seq_len=args.text_seq_len,
                      fourier_dim=args.text_fourier_dim, max_frequencies=args.text_max_freq)
        dm = (SST2PerceiverDataModule(**common) if task == "sst2"
              else GLUEPerceiverDataModule(task_name=task, **common))
        dm.setup()
        dms[task] = dm
    return dms


class MultitaskPerceiverIO(nn.Module):
    """Encoder+decoder condivisi, una query e una testa lineare per task."""

    def __init__(self, input_dim, num_classes_per_task, args):
        super().__init__()
        self.backbone = PerceiverIO(
            input_dim=input_dim,
            num_classes=1,                     # inutilizzato: le teste sono qui sotto
            num_latents=args.num_latents,
            latent_dim=args.latent_dim,
            num_cross_attend_stages=args.num_cross_attend_stages,
            num_transformer_blocks=args.num_transformer_blocks,
            num_heads=args.num_heads,
            head_dim=args.latent_dim // args.num_heads,
            dropout=args.dropout,
            num_output_queries=len(TASKS),     # una query per task
            task="features",
        )
        self.heads = nn.ModuleList(
            [nn.Linear(args.latent_dim, n) for n in num_classes_per_task]
        )

    def forward(self, x, task_idx):
        feats = self.backbone(x)               # [B, n_task, latent_dim]
        outputs = {}
        for t in task_idx.unique().tolist():
            sel = task_idx == t
            outputs[t] = self.heads[t](feats[sel, t])
        return outputs


def to_model_input(byte_ids, pos_encodings, device):
    """byte -> one-hot + positional encoding, come fanno i DataModule GLUE.
    Tutti i task condividono lo stesso PE, quindi una funzione sola basta."""
    x = torch.nn.functional.one_hot(byte_ids.to(device), num_classes=VOCAB).float()
    pe = pos_encodings[: x.shape[1]].to(device).unsqueeze(0).expand(x.shape[0], -1, -1)
    return torch.cat([x, pe], dim=-1)


def compute_loss(outputs, labels, task_idx, ce, mse):
    """Somma delle loss dei task presenti nel batch, pesata per numero di esempi."""
    total, n = 0.0, 0
    for t, out in outputs.items():
        sel = task_idx == t
        y = labels[sel]
        loss = mse(out.squeeze(-1), y.float()) if t == STSB else ce(out, y.long())
        total = total + loss * sel.sum()
        n += sel.sum()
    return total / max(1, n)


@torch.no_grad()
def evaluate(model, dms, pos_encodings, device):
    """Metrica per task sul dev set ufficiale: accuracy, Pearson per STS-B."""
    model.eval()
    scores = {}
    for t, task in enumerate(TASKS):
        preds, targs = [], []
        for batch in tqdm(dms[task].val_dataloader(), desc=f"eval {task}", leave=False):
            byte_ids, y = batch[0], batch[1]
            x = to_model_input(byte_ids, pos_encodings, device)
            idx = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
            out = model(x, idx)[t].float()
            preds.append((out.squeeze(-1) if t == STSB else out.argmax(1)).cpu())
            targs.append(y.cpu())
        p = torch.cat(preds).numpy()
        g = torch.cat(targs).numpy()
        if t == STSB:
            scores[task] = (float(np.corrcoef(p, g)[0, 1])
                            if p.std() > 0 and g.std() > 0 else 0.0)
        else:
            scores[task] = float((p == g).mean())
    return scores


def main():
    parser = argparse.ArgumentParser(description="Perceiver IO multitask su GLUE (Tab. 2)")
    parser.add_argument("--experiment_name", default="io_glue_multitask")
    parser.add_argument("--load_checkpoint_path", default=None,
                        help="checkpoint del pre-training MLM (senza, parte da zero)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--log_dir", default="logs")
    parser.add_argument("--num_latents", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--num_cross_attend_stages", type=int, default=1)
    parser.add_argument("--num_transformer_blocks", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--text_seq_len", type=int, default=512)
    parser.add_argument("--text_fourier_dim", type=int, default=64)
    parser.add_argument("--text_max_freq", type=float, default=64.0)
    parser.add_argument("--optimizer", default="lamb")
    parser.add_argument("--max_steps_per_epoch", type=int, default=4000,
                        help="i task grandi (QQP, MNLI) dominerebbero: si campiona")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    dms = build_datamodules(args)
    input_dim = dms["rte"].input_dim
    pos_encodings = dms["rte"].pos_encodings
    num_classes = [1 if t == "stsb" else dms[t].num_classes for t in TASKS]
    print(f"input_dim={input_dim}  classi per task={dict(zip(TASKS, num_classes))}")

    mixed = ConcatDataset([_Tagged(dms[t].train_dataset, i) for i, t in enumerate(TASKS)])
    print(f"esempi di training totali: {len(mixed):,}")
    loader = DataLoader(mixed, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, drop_last=True)

    model = MultitaskPerceiverIO(input_dim, num_classes, args).to(device)
    print(f"parametri: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    if args.load_checkpoint_path:
        if not os.path.exists(args.load_checkpoint_path):
            raise SystemExit(f"Checkpoint non trovato: {args.load_checkpoint_path}")
        sd = torch.load(args.load_checkpoint_path, map_location="cpu", weights_only=True)
        sd = sd.get("model_state_dict", sd) if isinstance(sd, dict) else sd
        target = model.backbone.state_dict()
        keep = {k: v for k, v in sd.items() if k in target and v.shape == target[k].shape}
        if not keep:
            raise SystemExit("Nessun peso compatibile: architettura diversa dal pre-training.")
        target.update(keep)
        model.backbone.load_state_dict(target, strict=False)
        print(f"Pre-training caricato: {len(keep)}/{len(target)} chiavi "
              f"(query e teste sono nuove, e' corretto che manchino)")

    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    # LAMB come in tutto il resto del progetto e come nel paper ("il Perceiver e'
    # piu' facile da ottimizzare con LAMB rispetto ad Adam"). Qui c'era AdamW
    # hardcoded: era l'unico punto del progetto che si scostava dalla ricetta, e
    # anche una delle due differenze fra MNLI single-task (65.80%) e MNLI dentro
    # il multitask (35.3%, cioe' il prior). Chiamata identica a train.py:385.
    if args.optimizer.lower() != "lamb":
        raise SystemExit(f"Ottimizzatore non previsto: {args.optimizer}")
    optimizer = custom_optim.Lamb(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = os.path.join(args.log_dir, args.experiment_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    best_avg, best_epoch, best_scores, final_avg = 0.0, 0, {}, 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        bar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, (byte_ids, y, task_idx) in enumerate(bar):
            if step >= args.max_steps_per_epoch:
                break
            x = to_model_input(byte_ids, pos_encodings, device)
            y, task_idx = y.to(device), task_idx.to(device)
            loss = compute_loss(model(x, task_idx), y, task_idx, ce, mse)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * y.size(0)
            seen += y.size(0)
            bar.set_postfix(loss=f"{running/max(1,seen):.4f}")
        scheduler.step()

        scores = evaluate(model, dms, pos_encodings, device)
        avg = sum(scores.values()) / len(scores)
        final_avg = avg
        print(f"Epoch {epoch}/{args.epochs}  media GLUE {avg*100:.2f}  " +
              "  ".join(f"{k}={v*100:.1f}" for k, v in scores.items()))
        if avg > best_avg:
            best_avg, best_epoch, best_scores = avg, epoch, scores
            torch.save(model.state_dict(), best_path)
            print(f"   nuovo best (epoca {epoch})")

    results = {
        "experiment": args.experiment_name,
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "val_accuracy": best_avg,          # media GLUE: confrontabile con Tab. 1/2
        "final_val_accuracy": final_avg,
        "test_accuracy": None,             # GLUE tiene nascoste le label di test
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "per_task": best_scores,
    }
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"\nmedia GLUE migliore: {best_avg*100:.2f} (epoca {best_epoch})")
    print(f"Results written to {os.path.join(run_dir, 'results.json')}")


if __name__ == "__main__":
    main()
