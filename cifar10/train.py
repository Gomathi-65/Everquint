"""Train a CIFAR-10 CNN.

    python train.py --model baseline --epochs 30 --amp
    python train.py --model residual --epochs 30 --amp
    python train.py --model baseline --fake --epochs 2        # 1-minute smoke test, no download
"""
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from data import get_loaders
from models import build_model, count_params
from utils import get_device, save_json, set_seed


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["baseline", "residual"], default="baseline")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=None, help="peak LR of the one-cycle schedule (default: 0.1 sgd, 0.003 adamw)")
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="runs")
    p.add_argument("--amp", action="store_true", help="mixed-precision training (CUDA only)")
    p.add_argument("--deterministic", action="store_true", help="bit-exact reproducibility (slower)")
    p.add_argument("--fake", action="store_true", help="random fake data, for smoke tests only")
    p.add_argument("--resume", action="store_true", help="continue from runs/<model>/last.pt")
    a = p.parse_args(argv)
    if a.lr is None:
        a.lr = 0.1 if a.optimizer == "sgd" else 0.003
    return a


def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None, amp=False):
    """One pass over the loader. Trains if an optimizer is given, otherwise evaluates. Returns (loss, accuracy)."""
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                out = model(x)
                loss = criterion(out, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()  # one-cycle schedule is stepped per batch
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main(argv=None):
    a = parse_args(argv)
    set_seed(a.seed, a.deterministic)
    device = get_device()
    amp = a.amp and device.type == "cuda"
    if a.amp and not amp:
        print("--amp needs a CUDA GPU; continuing in full precision")

    train_loader, val_loader, _ = get_loaders(a.data_dir, a.batch_size, num_workers=a.num_workers, seed=a.seed, fake=a.fake)
    model = build_model(a.model, dropout=a.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    if a.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=a.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=a.lr, epochs=a.epochs,
                                                    steps_per_epoch=len(train_loader), pct_start=0.25)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    run_dir = Path(a.out_dir) / a.model
    run_dir.mkdir(parents=True, exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    start_epoch, best_val = 0, 0.0
    if a.resume and (run_dir / "last.pt").exists():
        ck = torch.load(run_dir / "last.pt", map_location=device)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        history, start_epoch, best_val = ck["history"], ck["epoch"] + 1, ck["best_val_acc"]
        print(f"Resumed from epoch {start_epoch}")

    config = vars(a) | {"params": count_params(model), "device": str(device), "amp_used": amp}
    save_json(config, run_dir / "config.json")
    print(f"{a.model}: {config['params']:,} parameters on {device} | train {len(train_loader.dataset)} / val {len(val_loader.dataset)}")

    t0 = time.time()
    for epoch in range(start_epoch, a.epochs):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer, scheduler, scaler, amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, amp=amp)
        for k, v in zip(history, (tr_loss, tr_acc, va_loss, va_acc, optimizer.param_groups[0]["lr"])):
            history[k].append(v)
        if va_acc > best_val:
            best_val = va_acc
            torch.save({"model": model.state_dict(), "val_acc": va_acc, "epoch": epoch, "config": config}, run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                    "history": history, "epoch": epoch, "best_val_acc": best_val}, run_dir / "last.pt")
        save_json(history, run_dir / "history.json")
        print(f"epoch {epoch + 1:3d}/{a.epochs} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.3f} acc {va_acc:.3f} | lr {history['lr'][-1]:.4f} | {time.time() - t0:.0f}s")

    save_json({"best_val_acc": best_val, "train_seconds": time.time() - t0, "epochs": a.epochs}, run_dir / "summary.json")
    print(f"Done. Best validation accuracy {best_val:.4f}. Checkpoints in {run_dir}/")


if __name__ == "__main__":
    main()
