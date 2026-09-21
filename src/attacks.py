"""Poisoning attacks for controlled federated-learning experiments."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


def flip_labels(labels: np.ndarray, source_class: int | None = None, target_class: int | None = None) -> np.ndarray:
	"""Return copied labels with either binary inversion or targeted relabeling."""
	poisoned = labels.copy()
	if source_class is None:
		if not np.issubdtype(poisoned.dtype, np.integer):
			raise TypeError("untargeted label flipping requires integer labels")
		unique = np.unique(poisoned)
		if len(unique) != 2:
			raise ValueError("untargeted flipping requires exactly two classes")
		poisoned = np.where(poisoned == unique[0], unique[1], unique[0]).astype(labels.dtype)
	else:
		if target_class is None:
			raise ValueError("target_class is required for targeted flipping")
		poisoned[poisoned == source_class] = target_class
	return poisoned


def poison_labels(x: np.ndarray, y: np.ndarray, fraction: float, source_class: int | None = None, target_class: int | None = None, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
	"""Flip labels on a reproducible fraction of a local dataset."""
	if not 0 <= fraction <= 1:
		raise ValueError("fraction must be between 0 and 1")
	rng = np.random.default_rng(seed)
	count = int(round(len(y) * fraction))
	selected = rng.choice(len(y), size=count, replace=False)
	poisoned_y = y.copy()
	poisoned_y[selected] = flip_labels(poisoned_y[selected], source_class, target_class)
	return x.copy(), poisoned_y


def _clone_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
	return {name: value.detach().clone() for name, value in state.items()}


def _reference_delta(update: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
	return {name: update[name].detach() - reference[name].detach() for name in update}


def sign_flip(update: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], scale: float = 1.0) -> dict[str, torch.Tensor]:
	"""Return a model update reflected around the reference model."""
	if scale < 0:
		raise ValueError("scale must be non-negative")
	delta = _reference_delta(update, reference)
	return {name: reference[name].detach().clone() - scale * value for name, value in delta.items()}


def scale_update(update: Mapping[str, torch.Tensor], reference: Mapping[str, torch.Tensor], scale: float) -> dict[str, torch.Tensor]:
	"""Scale a client delta around the current global model."""
	if scale < 0:
		raise ValueError("scale must be non-negative")
	delta = _reference_delta(update, reference)
	return {name: reference[name].detach().clone() + scale * value for name, value in delta.items()}


def backdoor_trigger(x: np.ndarray, y: np.ndarray, fraction: float, target_class: int, feature_indices: list[int], trigger_value: float = 1.0, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
	"""Apply a feature trigger to selected samples and assign a target label."""
	if not feature_indices:
		raise ValueError("feature_indices must not be empty")
	if not 0 <= fraction <= 1:
		raise ValueError("fraction must be between 0 and 1")
	rng = np.random.default_rng(seed)
	poisoned_x, poisoned_y = x.copy(), y.copy()
	selected = rng.choice(len(y), size=int(round(len(y) * fraction)), replace=False)
	poisoned_x[np.ix_(selected, feature_indices)] = trigger_value
	poisoned_y[selected] = target_class
	return poisoned_x, poisoned_y
