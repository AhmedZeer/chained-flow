from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_log_history(output_dir: str | Path) -> list[dict[str, Any]]:
    state_path = Path(output_dir) / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing trainer state: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return list(state.get("log_history", []))


def loss_series(log_history: list[dict[str, Any]]) -> dict[str, tuple[list[float], list[float]]]:
    series = {
        "train": ([], []),
        "eval": ([], []),
    }
    for entry in log_history:
        epoch = entry.get("epoch")
        if not isinstance(epoch, int | float):
            continue
        if isinstance(entry.get("loss"), int | float):
            epochs, values = series["train"]
            epochs.append(float(epoch))
            values.append(float(entry["loss"]))
        if isinstance(entry.get("eval_loss"), int | float):
            epochs, values = series["eval"]
            epochs.append(float(epoch))
            values.append(float(entry["eval_loss"]))
    return series


def plot_flow_train_eval_loss(
    output_dir: str | Path,
    *,
    output_path: str | Path | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_path = Path(output_path) if output_path else output_dir / "flow_train_eval_loss.png"
    series = loss_series(load_log_history(output_dir))
    if not series["train"][0] and not series["eval"][0]:
        raise ValueError("no train loss or eval_loss entries found in trainer_state.json")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=False)
    for axis, key, title in (
        (axes[0], "train", "Train Loss"),
        (axes[1], "eval", "Eval Loss"),
    ):
        epochs, values = series[key]
        if epochs:
            axis.plot(epochs, values, marker="o", linewidth=1.75, markersize=3)
        else:
            axis.text(0.5, 0.5, f"no {key} loss logged", ha="center", va="center", transform=axis.transAxes)
        axis.set_title(title)
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.grid(True, alpha=0.25)

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot flow train loss and eval loss by epoch from trainer_state.json.")
    parser.add_argument("output_dir", help="Flow training output directory containing trainer_state.json.")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()
    path = plot_flow_train_eval_loss(args.output_dir, output_path=args.output_path)
    print(f"flow train/eval loss plot saved: {path}")


if __name__ == "__main__":
    main()
