"""Robust update screening, reputation, committee validation, and aggregators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def flatten_state(state: Mapping[str, torch.Tensor]) -> torch.Tensor:
	"""Flatten floating-point state tensors in deterministic key order."""
	return torch.cat([state[name].detach().float().reshape(-1) for name in sorted(state)])


def update_vector(update: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> torch.Tensor:
	return flatten_state({name: update[name] - reference[name] for name in update})


def update_norm(update: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> float:
	return float(torch.linalg.vector_norm(update_vector(update, reference)).item())


def cosine_similarity(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> float:
	left_vector = update_vector(left, reference)
	right_vector = update_vector(right, reference)
	denominator = torch.linalg.vector_norm(left_vector) * torch.linalg.vector_norm(right_vector)
	if denominator == 0:
		return 1.0 if torch.equal(left_vector, right_vector) else 0.0
	return float(torch.dot(left_vector, right_vector).div(denominator).item())


def similarity_norm_filter(updates: Sequence[Mapping[str, torch.Tensor]], reference: Mapping[str, torch.Tensor], similarity_threshold: float = 0.0, max_norm_multiplier: float = 2.0) -> list[int]:
	"""Return indices passing cosine-to-median and norm screening."""
	if not updates:
		return []
	vectors = torch.stack([update_vector(update, reference) for update in updates])
	median_vector = vectors.median(dim=0).values
	median_norm = float(torch.linalg.vector_norm(median_vector).item())
	norms = torch.linalg.vector_norm(vectors, dim=1)
	norm_limit = max(median_norm * max_norm_multiplier, 1e-12)
	passing: list[int] = []
	for index, vector in enumerate(vectors):
		denominator = torch.linalg.vector_norm(vector) * torch.linalg.vector_norm(median_vector)
		similarity = 1.0 if denominator == 0 else float(torch.dot(vector, median_vector).div(denominator).item())
		if similarity >= similarity_threshold and float(norms[index]) <= norm_limit:
			passing.append(index)
	return passing or [int(torch.argmin(norms).item())]


def update_reputations(reputations: Sequence[float], accepted: Sequence[bool], decay: float = 0.9) -> list[float]:
	"""Exponentially update client reputations from screening outcomes."""
	if len(reputations) != len(accepted) or not 0 <= decay < 1:
		raise ValueError("reputations and accepted must match; decay must be in [0, 1)")
	return [decay * float(score) + (1 - decay) * (1.0 if is_accepted else 0.0) for score, is_accepted in zip(reputations, accepted)]


def reputation_filter(reputations: Sequence[float], minimum: float = 0.5) -> list[int]:
	return [index for index, score in enumerate(reputations) if score >= minimum]


def committee_validate(model, x_val: torch.Tensor, y_val: torch.Tensor, candidate_states: Sequence[Mapping[str, torch.Tensor]], loss_tolerance: float = 0.0) -> list[int]:
	"""Accept candidates whose validation loss is no worse than the median plus tolerance."""
	criterion = torch.nn.CrossEntropyLoss()
	losses: list[float] = []
	for state in candidate_states:
		model.load_state_dict(state)
		model.eval()
		with torch.no_grad():
			losses.append(float(criterion(model(x_val), y_val.long()).item()))
	median_loss = float(torch.tensor(losses).median().item())
	return [index for index, loss in enumerate(losses) if loss <= median_loss + loss_tolerance]


def fedavg(updates: Sequence[Mapping[str, torch.Tensor]], sample_counts: Sequence[int]) -> dict[str, torch.Tensor]:
	if not updates or len(updates) != len(sample_counts) or sum(sample_counts) <= 0:
		raise ValueError("updates and positive sample_counts are required")
	total = sum(sample_counts)
	return {name: torch.stack([state[name].float() * count / total for state, count in zip(updates, sample_counts)]).sum(dim=0) for name in updates[0]}


def coordinate_median(updates: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
	if not updates:
		raise ValueError("at least one update is required")
	return {name: torch.stack([state[name].float() for state in updates]).median(dim=0).values for name in updates[0]}


def trimmed_mean(updates: Sequence[Mapping[str, torch.Tensor]], trim_fraction: float = 0.1) -> dict[str, torch.Tensor]:
	if not updates or not 0 <= trim_fraction < 0.5:
		raise ValueError("updates are required and trim_fraction must be in [0, 0.5)")
	tensor_count = len(updates)
	trim_count = min(int(tensor_count * trim_fraction), (tensor_count - 1) // 2)
	return {name: torch.stack([state[name].float() for state in updates]).sort(dim=0).values[trim_count:tensor_count - trim_count].mean(dim=0) for name in updates[0]}


def krum(updates: Sequence[Mapping[str, torch.Tensor]], malicious_count: int = 1) -> dict[str, torch.Tensor]:
	"""Select the update with the smallest sum of nearest-neighbor distances."""
	if not updates or malicious_count < 0 or len(updates) <= 2 * malicious_count + 2:
		raise ValueError("Krum requires more than 2f+2 client updates")
	vectors = torch.stack([flatten_state(state) for state in updates])
	distances = torch.cdist(vectors, vectors).pow(2)
	neighbor_count = len(updates) - malicious_count - 2
	scores = []
	for index in range(len(updates)):
		nearest = torch.topk(distances[index], k=neighbor_count + 1, largest=False).values[1:]
		scores.append(nearest.sum())
	return {name: value.detach().clone() for name, value in updates[int(torch.argmin(torch.stack(scores)).item())].items()}
