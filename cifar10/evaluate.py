"""Evaluate the best checkpoint on the test set; plot curves and misclassified images.

    python evaluate.py --model baseline
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from data import get_loaders  # noqa: E402
from models import build_model, count_params  # noqa: E402
from utils import CLASSES, MEAN, STD, get_device, load_json, save_json  # noqa: E402


def plot_curves(history, path, title):
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax[0].plot(epochs, history["train_loss"], label="train")
    ax[0].plot(epochs, history["val_loss"], label="validation")
    ax[0].set(title=f"{title}: loss", xlabel="epoch", ylabel="cross-entropy")
    ax[1].plot(epochs, history["train_acc"], label="train")
    ax[1].plot(epochs, history["val_acc"], label="validation")
    ax[1].set(title=f"{title}: accuracy", xlabel="epoch", ylabel="accuracy")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "residual"], default="baseline")
    p.add_argument("--run-dir", default="runs")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--fake", action="store_true")
    p.add_argument("--n-wrong", type=int, default=16)
    a = p.parse_args()

    device = get_device()
    run_dir = Path(a.run_dir) / a.model
    ck = torch.load(run_dir / "best.pt", map_location=device)
    cfg = ck["config"]
    model = build_model(a.model, dropout=cfg["dropout"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    _, _, test_loader = get_loaders(a.data_dir, cfg["batch_size"], num_workers=cfg["num_workers"], seed=cfg["seed"], fake=a.fake)

    crit = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, preds, targets, wrong_imgs, wrong_info = 0.0, [], [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += crit(out, y).item()
            pr = out.argmax(1)
            preds.append(pr.cpu())
            targets.append(y.cpu())
            for i in (pr != y).nonzero().flatten().tolist():
                if len(wrong_imgs) < a.n_wrong:
                    wrong_imgs.append(x[i].cpu())
                    wrong_info.append((y[i].item(), pr[i].item()))
    preds, targets = torch.cat(preds).numpy(), torch.cat(targets).numpy()
    acc = float((preds == targets).mean())

    per_class = {c: float((preds[targets == i] == i).mean()) if (targets == i).any() else None for i, c in enumerate(CLASSES)}
    cm = np.zeros((10, 10), dtype=int)
    np.add.at(cm, (targets, preds), 1)
    metrics = {"model": a.model, "test_accuracy": acc, "test_loss": loss_sum / len(targets), "params": count_params(model),
               "best_epoch": ck["epoch"] + 1, "best_val_acc": ck["val_acc"], "per_class_accuracy": per_class}
    save_json(metrics, run_dir / "test_metrics.json")
    print(f"{a.model}: test accuracy {acc:.4f} | test loss {metrics['test_loss']:.4f} | best epoch {metrics['best_epoch']}")
    for c, v in per_class.items():
        print(f"  {c:11s} {v if v is None else round(v, 3)}")

    plot_curves(load_json(run_dir / "history.json"), run_dir / "curves.png", a.model)

    fig, axes = plt.subplots(4, 4, figsize=(9, 9.5))
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, (t, pr) in zip(axes.flat, wrong_imgs, wrong_info):
        ax.imshow((img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
        ax.set_title(f"true: {CLASSES[t]}\npred: {CLASSES[pr]}", fontsize=8, color="firebrick")
    fig.suptitle(f"{a.model}: misclassified test images")
    fig.tight_layout()
    fig.savefig(run_dir / "misclassified.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true", title=f"{a.model}: confusion matrix")
    ax.set_xticklabels(CLASSES, rotation=60, ha="right", fontsize=8)
    ax.set_yticklabels(CLASSES, fontsize=8)
    fig.tight_layout()
    fig.savefig(run_dir / "confusion_matrix.png", dpi=130)
    plt.close(fig)
    print(f"Saved curves.png, misclassified.png, confusion_matrix.png, test_metrics.json in {run_dir}/")


if __name__ == "__main__":
    main()
