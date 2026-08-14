# bench.py
# Micro-benchmark: misura VRAM di picco e tempo per batch delle configurazioni candidate.
# Non addestra: gira 30 batch di forward+backward e estrapola.

import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from src.data.cifar10 import CIFAR10PerceiverDataModule
from src.perceiver.input_pe import InputPositionalEncoding
from src.perceiver.perceiver import Perceiver
from src.utils.seed import set_global_seed

CANDIDATES = [
    dict(name="base    T=4 L=4 N=96  D=384", N=96, D=384, T=4, L=4, batch=64),
    dict(name="T=8     T=8 L=4 N=96  D=384", N=96, D=384, T=8, L=4, batch=64),
    dict(name="T=12    T=12 L=4 N=96 D=384", N=96, D=384, T=12, L=4, batch=64),
    dict(name="media   T=8 L=6 N=256 D=512", N=256, D=512, T=8, L=6, batch=64),
    dict(name="fedele  T=8 L=6 N=512 D=1024", N=512, D=1024, T=8, L=6, batch=64),
]

BATCHES = 30
TRAIN_IMAGES = 45000


def benchmark(cfg, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    set_global_seed(42)

    pe = InputPositionalEncoding(grid_size=32, mode="fourier", num_bands=64, max_freq=16.0)
    model = Perceiver(
        token_dim=3, num_classes=10, input_pe=pe,
        num_latents=cfg["N"], latent_dim=cfg["D"],
        num_cross_attend_stages=cfg["T"], num_transformer_blocks=cfg["L"],
        num_heads_cross=1, num_heads_self=8, dropout=0.0,
    ).to(device)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler("cuda")

    x = torch.randn(cfg["batch"], 1024, 3, device=device)
    y = torch.randint(0, 10, (cfg["batch"],), device=device)

    for _ in range(3):  # warmup
        with autocast("cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(BATCHES):
        with autocast("cuda"):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    sec_per_batch = elapsed / BATCHES
    batches_per_epoch = TRAIN_IMAGES / cfg["batch"]
    minutes_per_epoch = sec_per_batch * batches_per_epoch / 60.0
    peak_gb = torch.cuda.max_memory_allocated() / 1024 ** 3

    del model, optimizer
    return params, peak_gb, minutes_per_epoch


def main():
    if not torch.cuda.is_available():
        raise SystemExit("serve una GPU")
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB)\n")
    print(f"{'config':34s} {'params':>12s} {'peak VRAM':>10s} {'min/epoca':>10s} {'ore/120ep':>10s}")
    print("-" * 82)

    for cfg in CANDIDATES:
        try:
            params, peak_gb, mins = benchmark(cfg, device)
            print(f"{cfg['name']:34s} {params:12,d} {peak_gb:9.2f}G {mins:10.2f} {mins*120/60:10.1f}")
        except torch.cuda.OutOfMemoryError:
            print(f"{cfg['name']:34s} {'-':>12s} {'OOM':>10s} {'-':>10s} {'-':>10s}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
