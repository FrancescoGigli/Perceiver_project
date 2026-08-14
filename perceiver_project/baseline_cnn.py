# baseline_cnn.py
# Baseline convoluzionale su CIFAR-10, il termine di paragone che manca al Perceiver.
# Il paper (Tab. 1) confronta il Perceiver con ResNet-50 e ViT su ImageNet: senza un
# baseline analogo, l'accuratezza del Perceiver da sola non dice se sia buona.
#
#   python baseline_cnn.py --experiment_name cnn_baseline
#
# Usa lo STESSO DataModule, lo stesso split di validazione e lo stesso seed delle run
# Perceiver, e scrive logs/<id>/results.json nello stesso formato: cosi' check.py e
# analyze_v2.py lo leggono senza modifiche.

import argparse
import json
import os

import torch
import torch.nn as nn
from torchvision.models import resnet18
from tqdm import tqdm

from src.data.cifar10 import CIFAR10PerceiverDataModule
from src.utils.seed import set_global_seed


def build_resnet18_for_cifar(num_classes=10):
    """ResNet-18 adattata a immagini 32x32: stem 3x3 e niente maxpool iniziale.
    Sulle immagini piccole lo stem 7x7 + maxpool di ImageNet butterebbe via subito
    troppa risoluzione (32 -> 8 prima ancora del primo blocco)."""
    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def run_epoch(model, loader, criterion, device, optimizer=None, desc=""):
    """Una passata. Con optimizer allena, senza valuta. Ritorna (loss, accuracy)."""
    train = optimizer is not None
    model.train(train)
    total_loss = total_correct = total = 0
    bar = tqdm(loader, desc=desc, leave=False)
    with torch.set_grad_enabled(train):
        for images, labels in bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * labels.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            bar.set_postfix(loss=f"{total_loss/total:.4f}", acc=f"{total_correct/total:.4f}")
    return total_loss / max(1, total), total_correct / max(1, total)


def main():
    parser = argparse.ArgumentParser(description="Baseline ResNet-18 su CIFAR-10")
    parser.add_argument("--experiment_name", default="cnn_baseline")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=int, default=5000)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--log_dir", default="logs")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Stesso DataModule delle run Perceiver: stesse trasformazioni, stesso split.
    dm = CIFAR10PerceiverDataModule(
        data_dir=args.data_dir, batch_size=args.batch_size,
        num_workers=args.num_workers, patch_size=1, val_split=args.val_split,
    )
    dm.setup()

    model = build_resnet18_for_cifar().to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResNet-18 (adattata a 32x32): {n_params:,} parametri")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                                weight_decay=args.weight_decay, nesterov=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_dir = os.path.join(args.log_dir, args.experiment_name)
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, "best_model.pt")

    best_val, best_epoch, final_val = 0.0, 0, 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, dm.train_dataloader(), criterion,
                                          device, optimizer, f"Epoch {epoch}/{args.epochs} [Train]")
        val_loss, val_acc = run_epoch(model, dm.val_dataloader(), criterion,
                                      device, None, f"Epoch {epoch}/{args.epochs} [Val]")
        scheduler.step()
        final_val = val_acc
        marker = ""
        if val_acc > best_val:
            best_val, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), best_path)
            marker = "  <- nuovo best"
        print(f"Epoch {epoch}/{args.epochs}  train {train_acc*100:.2f}%  "
              f"val {val_acc*100:.2f}%  (loss {val_loss:.4f}){marker}")

    # Il test set si tocca una volta sola, col checkpoint scelto sulla validation.
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    _, test_acc = run_epoch(model, dm.test_dataloader(), criterion, device, None, "Test")
    print(f"\nTEST accuracy: {test_acc*100:.2f}%  (checkpoint dell'epoca {best_epoch})")

    results = {
        "experiment": args.experiment_name,
        "seed": args.seed,
        "selected_epoch": best_epoch,
        "val_accuracy": best_val,
        "final_val_accuracy": final_val,
        "test_accuracy": test_acc,
        "params": n_params,
    }
    with open(os.path.join(run_dir, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    print(f"Results written to {os.path.join(run_dir, 'results.json')}")


if __name__ == "__main__":
    main()
