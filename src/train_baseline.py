"""Train and score the centralized MLP baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models import MLP, evaluate_model, model_size_kb, parameter_count, save_metrics, train_model

RESULT_COLUMNS = ["dataset", "model", "method", "attack", "malicious_fraction", "alpha", "seed", "round", "accuracy", "macro_f1", "precision", "recall", "fpr", "params", "model_kb", "bytes_per_round", "round_time_s"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--dataset-name", default="primary")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    processed = Path(config["processed_dir"])
    x_train = np.load(processed / "X_train.npy"); x_test = np.load(processed / "X_test.npy"); y_train = np.load(processed / "y_train.npy"); y_test = np.load(processed / "y_test.npy")
    torch.manual_seed(config["seed"])
    model = MLP(x_train.shape[1], len(json.loads((processed / "metadata.json").read_text())["class_mapping"]), config["hidden_dims"])
    seconds = train_model(model, x_train, y_train, config["epochs"], config["batch_size"], config["learning_rate"], config["weight_decay"], config["seed"])
    metrics = evaluate_model(model, x_test, y_test)
    results_dir = Path("results"); results_dir.mkdir(exist_ok=True)
    save_metrics(metrics, results_dir / f"{args.dataset_name}_centralized_metrics.json")
    torch.save(model.state_dict(), results_dir / f"{args.dataset_name}_centralized.pt")
    row = {"dataset": args.dataset_name, "model": "mlp", "method": "centralized", "attack": "none", "malicious_fraction": "NA", "alpha": "NA", "seed": config["seed"], "round": "NA", **{key: metrics[key] for key in ("accuracy", "macro_f1", "precision", "recall", "fpr")}, "params": parameter_count(model), "model_kb": round(model_size_kb(model), 3), "bytes_per_round": "NA", "round_time_s": round(seconds, 3)}
    with (results_dir / f"{args.dataset_name}_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS); writer.writeheader(); writer.writerow(row)
    print(json.dumps({key: row[key] for key in ("accuracy", "macro_f1", "precision", "recall", "fpr", "params", "model_kb", "round_time_s")}, indent=2))


if __name__ == "__main__":
    main()
