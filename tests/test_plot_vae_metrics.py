import importlib.util
import json
from pathlib import Path


def load_script_module():
    path = Path("scripts/plot_vae_metrics.py").resolve()
    spec = importlib.util.spec_from_file_location("plot_vae_metrics", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_metric_series_extracts_numeric_step_metrics():
    module = load_script_module()
    series = module.metric_series(
        [
            {"step": 1, "loss": 3.0, "epoch": 0.1},
            {"step": 2, "eval_loss": 2.0, "train_runtime": 10.0},
        ]
    )

    assert series == {"loss": ([1], [3.0]), "eval_loss": ([2], [2.0])}


def test_load_log_history_reads_trainer_state(tmp_path):
    module = load_script_module()
    state_path = tmp_path / "trainer_state.json"
    state_path.write_text(json.dumps({"log_history": [{"step": 1, "loss": 1.0}]}), encoding="utf-8")

    assert module.load_log_history(tmp_path) == [{"step": 1, "loss": 1.0}]
