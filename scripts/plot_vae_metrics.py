from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


IGNORED_KEYS = {
    "epoch",
    "step",
    "total_flos",
    "train_runtime",
    "train_samples_per_second",
    "train_steps_per_second",
}


def load_log_history(output_dir: str | Path) -> list[dict[str, Any]]:
    state_path = Path(output_dir) / "trainer_state.json"
    if not state_path.exists():
        raise FileNotFoundError(f"missing trainer state: {state_path}")
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    return list(state.get("log_history", []))


def metric_series(log_history: list[dict[str, Any]]) -> dict[str, tuple[list[int], list[float]]]:
    series: dict[str, tuple[list[int], list[float]]] = {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        for name, value in entry.items():
            if name in IGNORED_KEYS or not isinstance(value, int | float):
                continue
            steps, values = series.setdefault(name, ([], []))
            steps.append(int(step))
            values.append(float(value))
    return series


def plot_metrics(
    output_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    metrics: list[str] | None = None,
) -> Path:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_path = Path(output_path) if output_path else output_dir / "vae_metrics.png"
    series = metric_series(load_log_history(output_dir))
    selected = metrics or sorted(series)
    selected = [name for name in selected if name in series]
    if not selected:
        raise ValueError("no matching numeric metrics found in trainer_state.json")

    figure, axis = plt.subplots(figsize=(10, 6))
    for name in selected:
        steps, values = series[name]
        axis.plot(steps, values, marker="o", linewidth=1.5, markersize=3, label=name)
    axis.set_xlabel("step")
    axis.set_ylabel("metric")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best", fontsize="small")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot VAE Trainer metrics from trainer_state.json.")
    parser.add_argument("output_dir")
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--metrics", nargs="*", default=None)
    args = parser.parse_args()
    path = plot_metrics(args.output_dir, output_path=args.output_path, metrics=args.metrics)
    print(f"VAE metric plot saved: {path}")


if __name__ == "__main__":
    main()
