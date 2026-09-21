"""Clean PyTorch FedAvg with Dirichlet non-IID client partitions."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import MLP, evaluate_model, model_size_kb, parameter_count
from src.results import append_result


def dirichlet_partitions(labels: np.ndarray, client_count: int, alpha: float, seed: int) -> list[np.ndarray]:
	"""Partition indices by class using a reproducible Dirichlet allocation."""
	if client_count < 2 or alpha <= 0:
		raise ValueError("client_count must be at least 2 and alpha must be positive")
	rng = np.random.default_rng(seed)
	partitions = [[] for _ in range(client_count)]
	for class_id in np.unique(labels):
		class_indices = np.flatnonzero(labels == class_id)
		rng.shuffle(class_indices)
		proportions = rng.dirichlet(np.full(client_count, alpha, dtype=float))
		cut_points = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
		for client_id, client_indices in enumerate(np.split(class_indices, cut_points)):
			partitions[client_id].extend(client_indices.tolist())
	result = [np.asarray(indices, dtype=np.int64) for indices in partitions]
	if any(len(indices) == 0 for indices in result):
		return dirichlet_partitions(labels, client_count, alpha, seed + 1)
	return result


def _state_bytes(state: dict[str, torch.Tensor]) -> int:
	buffer = io.BytesIO()
	torch.save(state, buffer)
	return buffer.getbuffer().nbytes


def _local_update(global_state, x_train, y_train, indices, model_factory, config, seed):
	model = model_factory()
	model.load_state_dict(global_state)
	torch.manual_seed(seed)
	loader = DataLoader(TensorDataset(torch.from_numpy(x_train[indices]), torch.from_numpy(y_train[indices]).long()), batch_size=int(config["batch_size"]), shuffle=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
	criterion = nn.CrossEntropyLoss()
	model.train()
	for _ in range(int(config["local_epochs"])):
		for features, labels in loader:
			optimizer.zero_grad(set_to_none=True)
			criterion(model(features), labels).backward()
			optimizer.step()
	return {name: value.detach().clone() for name, value in model.state_dict().items()}, len(indices)


def _fedavg(states, sample_counts):
	total_samples = sum(sample_counts)
	return {name: torch.stack([state[name].float() * count / total_samples for state, count in zip(states, sample_counts)]).sum(dim=0) for name in states[0]}


def run_fedavg(x_train, y_train, x_test, y_test, input_dim, class_count, config, dataset_name, client_count, alpha, output_path):
	"""Run FedAvg and append one standardized metrics row per global round."""
	seed = int(config["seed"])
	partitions = dirichlet_partitions(y_train, client_count, alpha, seed + client_count * 1000 + round(alpha * 100))
	model_factory = lambda: MLP(input_dim, class_count, config["hidden_dims"])
	model = model_factory()
	global_state = model.state_dict()
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	bytes_per_round = _state_bytes(global_state) * (client_count + 1)
	rows = []
	for round_number in range(1, int(config["rounds"]) + 1):
		started = time.perf_counter()
		client_states, sample_counts = [], []
		for client_id, indices in enumerate(partitions):
			state, count = _local_update(global_state, x_train, y_train, indices, model_factory, config, seed + round_number * 10000 + client_id)
			client_states.append(state)
			sample_counts.append(count)
		global_state = _fedavg(client_states, sample_counts)
		model.load_state_dict(global_state)
		metrics = evaluate_model(model, x_test, y_test)
		row = {"dataset": dataset_name, "model": "mlp", "method": "fedavg", "clients": client_count, "attack": "none", "malicious_fraction": 0.0, "alpha": alpha, "seed": seed, "round": round_number, **{key: metrics[key] for key in ("accuracy", "macro_f1", "precision", "recall", "fpr")}, "params": parameter_count(model), "model_kb": round(model_size_kb(model), 3), "bytes_per_round": bytes_per_round, "round_time_s": round(time.perf_counter() - started, 3)}
		append_result(output_path, row)
		rows.append(row)
		print(json.dumps({"clients": client_count, "alpha": alpha, "round": round_number, "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "fpr": row["fpr"]}))
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/baseline.yaml")
	parser.add_argument("--dataset-name", default="edge_iiotset")
	parser.add_argument("--clients", type=int, nargs="+", default=None)
	parser.add_argument("--alphas", type=float, nargs="+", default=None)
	parser.add_argument("--output", default="results/edge_iiotset_fedavg_results.csv")
	args = parser.parse_args()
	config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
	processed = Path(config["processed_dir"])
	x_train = np.load(processed / "X_train.npy"); y_train = np.load(processed / "y_train.npy")
	x_test = np.load(processed / "X_test.npy"); y_test = np.load(processed / "y_test.npy")
	metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
	clients = args.clients or config.get("client_counts", [10, 20])
	alphas = args.alphas or config.get("alphas", [0.1, 0.5, 1.0])
	output = Path(args.output)
	if output.exists():
		output.unlink()
	for client_count in clients:
		for alpha in alphas:
			run_fedavg(x_train, y_train, x_test, y_test, x_train.shape[1], len(metadata["class_mapping"]), config, args.dataset_name, client_count, alpha, output)


if __name__ == "__main__":
	main()
