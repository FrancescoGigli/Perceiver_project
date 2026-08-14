# visualize_v2_attention.py
# Estrae e visualizza le mappe di cross-attention di un Perceiver v2 gia'
# allenato (analogo della Fig. 3 del paper): per alcune immagini del test set
# CIFAR-10, mostra dove guardano i latenti nell'ULTIMO stage di cross-attention.
#
# Sola lettura: carica config.txt + checkpoint di una run, ricostruisce
# l'architettura esatta e fa un forward. Non lancia training, non tocca
# src/, train.py o experiments.py. Girato su CPU (map_location='cpu') per non
# competere con eventuali run in corso sulla GPU.
#
#   python visualize_v2_attention.py --experiment e01_baseline [--num-images 4] [--out perceiver_visualizations_v2]

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.cifar10 import CIFAR10PerceiverDataModule
from src.perceiver.input_pe import InputPositionalEncoding, make_token_permutation
from src.perceiver.perceiver import Perceiver

CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)


def _convert(value):
    """Converte una stringa di config.txt nel tipo Python corrispondente."""
    if value == "True":
        return True
    if value == "False":
        return False
    if value == "None":
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_config(path):
    """Legge la sezione 'COMMAND LINE ARGUMENTS:' di logs/<experiment>/config.txt.

    Il formato e' 'chiave: valore' per riga, delimitato da una riga di trattini
    subito dopo l'intestazione e terminato dalla prima riga vuota.
    """
    with open(path, encoding="utf-8") as handle:
        lines = handle.readlines()

    start = None
    for i, line in enumerate(lines):
        if line.strip() == "COMMAND LINE ARGUMENTS:":
            start = i + 1
            break
    if start is None:
        raise ValueError(f"sezione 'COMMAND LINE ARGUMENTS:' non trovata in {path}")

    args = {}
    for line in lines[start:]:
        line = line.rstrip("\n")
        if line.startswith("---"):
            continue
        if not line.strip():
            break  # fine della sezione
        if ":" not in line:
            break
        key, _, value = line.partition(":")
        args[key.strip()] = _convert(value.strip())
    return args


def build_model(cfg):
    """Ricostruisce l'architettura Perceiver esatta usata per allenare il checkpoint.

    save_attention_maps=True e' essenziale: senza, model.attn_maps resta vuoto.
    """
    patch_size = cfg["patch_size"]
    grid = 32 // patch_size

    if cfg.get("no_positional_encoding"):
        mode = "none"
    elif cfg.get("use_learned_pe"):
        mode = "learned"
    else:
        mode = "fourier"

    perm = None
    if cfg.get("permute_pixels"):
        perm = make_token_permutation(grid * grid, cfg["permute_pixels_seed"])

    input_pe = InputPositionalEncoding(
        grid_size=grid,
        mode=mode,
        num_bands=cfg["fourier_num_bands"],
        max_freq=cfg["fourier_max_freq"],
        learned_dim=cfg.get("learned_pe_dim", 128),
        init_scale=cfg["latent_init_scale"],
        permutation=perm,
    )

    model = Perceiver(
        token_dim=patch_size * patch_size * 3,
        num_classes=10,
        input_pe=input_pe,
        num_latents=cfg["num_latents"],
        latent_dim=cfg["latent_dim"],
        num_cross_attend_stages=cfg["num_cross_attend_stages"],
        num_transformer_blocks=cfg["num_transformer_blocks"],
        num_heads_cross=cfg["num_heads_cross"],
        num_heads_self=cfg["num_heads_self"],
        dropout=0.0,
        latent_init_scale=cfg["latent_init_scale"],
        save_attention_maps=True,
        weight_sharing=not cfg["no_weight_sharing"],
        arrangement=cfg["cross_attend_arrangement"],
        use_latent_transformer=not cfg["no_latent_transformer"],
        share_cross_attend=not cfg["no_share_cross_attend"],
    )
    return model, patch_size


def load_model(experiment, log_dir="logs"):
    """Carica config + checkpoint di una run e ricostruisce il modello (CPU, eval)."""
    exp_dir = os.path.join(log_dir, experiment)
    config_path = os.path.join(exp_dir, "config.txt")
    ckpt_path = os.path.join(exp_dir, "checkpoints", "best_model.pt")

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config non trovato: {config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint non trovato: {ckpt_path}")

    cfg = parse_config(config_path)
    model, patch_size = build_model(cfg)

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model, cfg, patch_size


def get_batch(cfg, patch_size, num_images):
    """Prime `num_images` immagini del test set CIFAR-10 (nessuno shuffle -> deterministico)."""
    data_dir = cfg.get("data_dir") or "./data"
    dm = CIFAR10PerceiverDataModule(
        data_dir=data_dir,
        batch_size=num_images,
        num_workers=0,
        patch_size=patch_size,
        val_split=cfg.get("val_split", 5000),
    )
    dm.setup()
    batch = next(iter(dm.test_dataloader()))
    processed = dm.preprocess_batch(batch)
    inputs = processed["inputs"][:num_images]
    original_images = processed["original_images"][:num_images]
    labels = processed["labels"][:num_images]
    return inputs, original_images, labels


def unnormalize(img):
    """[3, H, W] normalizzato -> numpy [H, W, 3] in [0, 1], per la visualizzazione."""
    img = img * CIFAR10_STD + CIFAR10_MEAN
    img = img.clamp(0.0, 1.0)
    return img.permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser(
        description="Visualizza le mappe di cross-attention di un Perceiver v2 allenato (stile Fig. 3 del paper)."
    )
    parser.add_argument(
        "--experiment", required=True,
        help="id della run, es. e01_baseline (deve esistere logs/<experiment>/checkpoints/best_model.pt)",
    )
    parser.add_argument("--num-images", type=int, default=4, help="numero di immagini del test set da visualizzare")
    parser.add_argument("--out", default="perceiver_visualizations_v2", help="cartella di output per i PNG")
    parser.add_argument(
        "--num-latents-grid", type=int, default=8,
        help="numero di singoli latenti mostrati nella griglia in stile Fig. 3",
    )
    args = parser.parse_args()

    if args.num_images < 1:
        raise SystemExit("--num-images deve essere >= 1")

    os.makedirs(args.out, exist_ok=True)

    print(f"Carico config + checkpoint per '{args.experiment}' (CPU)...")
    model, cfg, patch_size = load_model(args.experiment)
    print(
        f"Modello ricostruito: latent_dim={cfg['latent_dim']} num_latents={cfg['num_latents']} "
        f"T={cfg['num_cross_attend_stages']} arrangement={cfg['cross_attend_arrangement']} "
        f"patch_size={patch_size}"
    )

    inputs, original_images, labels = get_batch(cfg, patch_size, args.num_images)
    print(f"Batch di test caricato: {inputs.shape[0]} immagini, inputs {tuple(inputs.shape)}")

    with torch.no_grad():
        model(inputs)

    if not model.attn_maps:
        raise RuntimeError(
            "model.attn_maps e' vuoto dopo il forward: il modello non ha salvato le mappe di "
            "cross-attention. Verifica che save_attention_maps=True sia passato al costruttore "
            "di Perceiver (in questo script e' impostato in build_model())."
        )

    last_stage = model.attn_maps[-1]  # [B, num_heads_cross, N, M]
    print(f"attn_maps: {len(model.attn_maps)} stage(s) salvati, ultimo stage shape {tuple(last_stage.shape)}")

    num_tokens = inputs.shape[1]
    grid_side = int(round(num_tokens ** 0.5))
    if grid_side * grid_side != num_tokens:
        raise RuntimeError(
            f"il numero di token M={num_tokens} non e' un quadrato perfetto: impossibile fare reshape a griglia 2D"
        )

    num_images = inputs.shape[0]
    num_latents_total = last_stage.shape[2]
    num_latent_maps = max(1, min(args.num_latents_grid, num_latents_total))

    saved_paths = []
    for img_idx in range(num_images):
        original = unnormalize(original_images[img_idx])

        per_head = last_stage[img_idx].mean(dim=0)      # [N, M]  media sulle head di cross-attention
        mean_over_latents = per_head.mean(dim=0)         # [M]     media sui latenti
        heatmap = mean_over_latents.reshape(grid_side, grid_side).numpy()

        # --- immagine originale + heatmap media (ultimo stage) ---
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].imshow(original)
        axes[0].set_title(f"originale (label {int(labels[img_idx])})")
        axes[0].axis("off")
        im = axes[1].imshow(heatmap, cmap="viridis")
        axes[1].set_title(f"attenzione media, stage {len(model.attn_maps) - 1}")
        axes[1].axis("off")
        fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        fig.suptitle(f"{args.experiment} - immagine {img_idx}")
        fig.tight_layout()

        path = os.path.join(args.out, f"{args.experiment}_img{img_idx}_mean_attention.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(path)

        # --- griglia dei singoli latenti (stile Fig. 3 del paper) ---
        ncols = min(4, num_latent_maps)
        nrows = -(-num_latent_maps // ncols)  # ceil division
        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows), squeeze=False)
        axes2 = axes2.reshape(-1)
        for latent_idx in range(num_latent_maps):
            latent_map = per_head[latent_idx].reshape(grid_side, grid_side).numpy()
            axes2[latent_idx].imshow(latent_map, cmap="viridis")
            axes2[latent_idx].set_title(f"latente {latent_idx}", fontsize=9)
            axes2[latent_idx].axis("off")
        for extra_idx in range(num_latent_maps, len(axes2)):
            axes2[extra_idx].axis("off")
        fig2.suptitle(
            f"{args.experiment} - immagine {img_idx} - primi {num_latent_maps} latenti (ultimo stage)"
        )
        fig2.tight_layout()

        path2 = os.path.join(args.out, f"{args.experiment}_img{img_idx}_latents_grid.png")
        fig2.savefig(path2, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        saved_paths.append(path2)

    print(f"\n{len(saved_paths)} PNG salvati in '{args.out}':")
    for path in saved_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
