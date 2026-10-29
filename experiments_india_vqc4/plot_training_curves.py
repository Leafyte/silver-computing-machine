"""Plot observed train-MSE and model-selection MAE from the frozen epoch log."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "artifacts" / "india_eta_vqc4_run2"


def main() -> None:
    rows = [json.loads(line) for line in (RUN / "training_progress.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError("training log is empty")
    csv_path = RUN / "training_curves.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("stage", "epoch", "train_mse_min2", "selection_mae_min", "selection_rmse_min", "epoch_seconds", "optimizer_steps", "device"))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "stage": row["stage"], "epoch": row["epoch"], "train_mse_min2": row["train_mse_min2"],
                "selection_mae_min": row["selection"]["mae_min"], "selection_rmse_min": row["selection"]["rmse_min"],
                "epoch_seconds": row["epoch_seconds"], "optimizer_steps": row["optimizer_steps"], "device": row["device"],
            })
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {"baseline60": "#2867a8", "extended76": "#d16b24"}
    for stage in ("baseline60", "extended76"):
        subset = [row for row in rows if row["stage"] == stage]
        epoch = [row["epoch"] for row in subset]
        axes[0].plot(epoch, [row["train_mse_min2"] for row in subset], marker="o", label=stage, color=colors[stage])
        axes[1].plot(epoch, [row["selection"]["mae_min"] for row in subset], marker="o", label=stage, color=colors[stage])
    axes[0].set_title("Observed training MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (minutes²)")
    axes[0].grid(alpha=0.25)
    axes[1].set_title("Courier-disjoint model-selection MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE (minutes)")
    axes[1].grid(alpha=0.25)
    for ax in axes:
        ax.legend(frameon=False)
    fig.suptitle("India ETA VQC training — actual epoch logs")
    fig.savefig(RUN / "training_curves.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
