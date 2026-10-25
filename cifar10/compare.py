"""Baseline vs residual comparison table + overlaid curves.   python compare.py"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from utils import load_json  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--run-dir", default="runs")
p.add_argument("--out", default="results")
a = p.parse_args()
out = Path(a.out)
out.mkdir(exist_ok=True)

runs = {}
for name in ("baseline", "residual"):
    d = Path(a.run_dir) / name
    if not (d / "test_metrics.json").exists():
        raise SystemExit(f"Missing {d}/test_metrics.json. Train and evaluate '{name}' first.")
    runs[name] = {"m": load_json(d / "test_metrics.json"), "h": load_json(d / "history.json"), "s": load_json(d / "summary.json")}

rows = [("Parameters", lambda r: f"{r['m']['params']:,}"),
        ("Epochs", lambda r: str(r["s"]["epochs"])),
        ("Best validation accuracy", lambda r: f"{r['m']['best_val_acc']:.4f}"),
        ("Test accuracy", lambda r: f"{r['m']['test_accuracy']:.4f}"),
        ("Test loss", lambda r: f"{r['m']['test_loss']:.4f}"),
        ("Training time (s)", lambda r: f"{r['s']['train_seconds']:.0f}")]
lines = ["| Metric | Baseline CNN | + Residual blocks |", "|---|---|---|"]
lines += [f"| {n} | {f(runs['baseline'])} | {f(runs['residual'])} |" for n, f in rows]
diff = runs["residual"]["m"]["test_accuracy"] - runs["baseline"]["m"]["test_accuracy"]
lines.append(f"\nTest accuracy difference (residual - baseline): {diff * 100:+.2f} points (single run, seed identical).")
(out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for name, r in runs.items():
    e = range(1, len(r["h"]["val_loss"]) + 1)
    ax[0].plot(e, r["h"]["val_loss"], label=name)
    ax[1].plot(e, r["h"]["val_acc"], label=name)
ax[0].set(title="Validation loss", xlabel="epoch")
ax[1].set(title="Validation accuracy", xlabel="epoch")
for x in ax:
    x.grid(alpha=0.3)
    x.legend()
fig.tight_layout()
fig.savefig(out / "comparison_curves.png", dpi=130)
print(f"Saved {out}/comparison.md and comparison_curves.png")
