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

from src.attacks import adaptive_stealth, backdoor_trigger, poison_labels, scale_update, sign_flip
from src.crypto import generate_signing_key, hash_state, sign_payload
from src.defense import committee_validate, coordinate_median, fedavg, krum, reputation_filter, similarity_norm_filter, trimmed_mean, update_reputations
from src.ledger import FabricLedger, SimulatedLedger
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


def _local_update(global_state, x_train, y_train, indices, model_factory, config, seed, attack, malicious):
	model = model_factory()
	model.load_state_dict(global_state)
	torch.manual_seed(seed)
	local_x, local_y = x_train[indices], y_train[indices]
	if malicious and attack == "label_flip":
		_, local_y = poison_labels(local_x, local_y, float(config["attack_fraction"]), int(config["label_flip_source"]), int(config["label_flip_target"]), seed)
	elif malicious and attack == "backdoor":
		local_x, local_y = backdoor_trigger(local_x, local_y, float(config["attack_fraction"]), int(config["backdoor_target"]), list(config["backdoor_features"]), float(config["backdoor_value"]), seed)
	loader = DataLoader(TensorDataset(torch.from_numpy(local_x), torch.from_numpy(local_y).long()), batch_size=int(config["batch_size"]), shuffle=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
	criterion = nn.CrossEntropyLoss()
	model.train()
	for _ in range(int(config["local_epochs"])):
		for features, labels in loader:
			optimizer.zero_grad(set_to_none=True)
			criterion(model(features), labels).backward()
			optimizer.step()
	state = {name: value.detach().clone() for name, value in model.state_dict().items()}
	if malicious and attack == "sign_flip":
		state = sign_flip(state, global_state, float(config["attack_scale"]))
	elif malicious and attack == "scaling":
		state = scale_update(state, global_state, float(config["attack_scale"]))
	elif malicious and attack == "adaptive":
		state = adaptive_stealth(state, global_state, float(config["attack_scale"]), float(config.get("max_norm_multiplier", 1.95)))
	return state, len(indices)


def run_fedavg(x_train, y_train, x_val, y_val, x_test, y_test, input_dim, class_count, config, dataset_name, client_count, alpha, output_path, attack="none", defense="fedavg", ledger_type="simulated", custom_seed=None):
	"""Run FedAvg and append one standardized metrics row per global round."""
	seed = int(custom_seed if custom_seed is not None else config["seed"])
	partitions = dirichlet_partitions(y_train, client_count, alpha, seed + client_count * 1000 + round(alpha * 100))
	model_factory = lambda: MLP(input_dim, class_count, config["hidden_dims"])
	model = model_factory()
	global_state = model.state_dict()
	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	bytes_per_round = _state_bytes(global_state) * (client_count + 1)
	reputations = [1.0] * client_count
	rows = []

	keys = [generate_signing_key() for _ in range(client_count)]
	if ledger_type == "fabric":
		ledger = FabricLedger(Path("/tmp/blockfed-ledger.json"))
	else:
		ledger = SimulatedLedger(Path("/tmp/blockfed-ledger.json"))

	for round_number in range(1, int(config["rounds"]) + 1):
		started = time.perf_counter()
		client_states, sample_counts = [], []
		for client_id, indices in enumerate(partitions):
			malicious = client_id < int(round(client_count * float(config["malicious_fraction"])))
			state, count = _local_update(global_state, x_train, y_train, indices, model_factory, config, seed + round_number * 10000 + client_id, attack, malicious)
			client_states.append(state)
			sample_counts.append(count)

			priv_key, pub_key = keys[client_id]
			u_hash = hash_state(state)
			payload = {"round": round_number, "client_id": f"client-{client_id}", "update_hash": u_hash}
			sig = sign_payload(priv_key, payload)
			ledger.append_update(f"client-{client_id}", round_number, u_hash, payload, sig, pub_key)

		accepted = list(range(client_count))
		if defense == "similarity_norm":
			accepted = similarity_norm_filter(client_states, global_state, float(config["similarity_threshold"]), float(config["max_norm_multiplier"]))
		elif defense == "reputation":
			accepted = similarity_norm_filter(client_states, global_state, float(config["similarity_threshold"]), float(config["max_norm_multiplier"]))
			reputations = update_reputations(reputations, [index in accepted for index in range(client_count)], float(config["reputation_decay"]))
			accepted = reputation_filter(reputations, float(config["reputation_minimum"])) or accepted
		elif defense == "committee":
			accepted = committee_validate(model, torch.from_numpy(x_val), torch.from_numpy(y_val), client_states, float(config["committee_loss_tolerance"]))
		filtered_states = [client_states[index] for index in accepted]
		filtered_counts = [sample_counts[index] for index in accepted]
		if defense in {"median", "trimmed_mean", "krum"}:
			if defense == "median":
				global_state = coordinate_median(filtered_states)
			elif defense == "trimmed_mean":
				global_state = trimmed_mean(filtered_states, float(config["trim_fraction"]))
			else:
				global_state = krum(filtered_states, min(int(config["krum_malicious_count"]), max((len(filtered_states) - 3) // 2, 0)))
		else:
			global_state = fedavg(filtered_states, filtered_counts)
		model.load_state_dict(global_state)
		metrics = evaluate_model(model, x_test, y_test)
		row = {"dataset": dataset_name, "model": "mlp", "method": defense, "clients": client_count, "attack": attack, "malicious_fraction": float(config["malicious_fraction"]), "alpha": alpha, "seed": seed, "round": round_number, **{key: metrics[key] for key in ("accuracy", "macro_f1", "precision", "recall", "fpr")}, "params": parameter_count(model), "model_kb": round(model_size_kb(model), 3), "bytes_per_round": bytes_per_round, "round_time_s": round(time.perf_counter() - started, 3)}
		append_result(output_path, row)
		rows.append(row)
		print(json.dumps({"clients": client_count, "alpha": alpha, "seed": seed, "round": round_number, "accuracy": row["accuracy"], "macro_f1": row["macro_f1"], "fpr": row["fpr"]}))
	return rows


def main() -> None:
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="configs/baseline.yaml")
	parser.add_argument("--dataset-name", default="edge_iiotset")
	parser.add_argument("--clients", type=int, nargs="+", default=None)
	parser.add_argument("--alphas", type=float, nargs="+", default=None)
	parser.add_argument("--seeds", type=int, nargs="+", default=None)
	parser.add_argument("--output", default="results/edge_iiotset_fedavg_results.csv")
	parser.add_argument("--attack", default="none", choices=["none", "label_flip", "sign_flip", "scaling", "backdoor", "adaptive"])
	parser.add_argument("--defense", default="fedavg", choices=["fedavg", "similarity_norm", "reputation", "committee", "median", "trimmed_mean", "krum"])
	parser.add_argument("--ledger", default="simulated", choices=["simulated", "fabric"])
	args = parser.parse_args()
	config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
	processed = Path(config["processed_dir"])
	x_train = np.load(processed / "X_train.npy"); y_train = np.load(processed / "y_train.npy")
	x_val = np.load(processed / "X_val.npy"); y_val = np.load(processed / "y_val.npy"); x_test = np.load(processed / "X_test.npy"); y_test = np.load(processed / "y_test.npy")
	metadata = json.loads((processed / "metadata.json").read_text(encoding="utf-8"))
	clients = args.clients or config.get("client_counts", [10, 20])
	alphas = args.alphas or config.get("alphas", [0.1, 0.5, 1.0])
	seeds = args.seeds or [int(config["seed"])]
	output = Path(args.output)
	if output.exists():
		output.unlink()
	for s in seeds:
		for client_count in clients:
			for alpha in alphas:
				run_fedavg(x_train, y_train, x_val, y_val, x_test, y_test, x_train.shape[1], len(metadata["class_mapping"]), config, args.dataset_name, client_count, alpha, output, args.attack, args.defense, args.ledger, custom_seed=s)


if __name__ == "__main__":
	main()
