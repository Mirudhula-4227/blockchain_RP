"""Differential Privacy (DP-FedAvg) noise addition and update clipping module."""

from __future__ import annotations

from typing import Mapping

import torch


def clip_and_add_noise(
    update: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
    clip_norm: float = 1.0,
    noise_multiplier: float = 0.01,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Clip client delta to max L2 norm and add calibrated Gaussian noise for Differential Privacy."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be positive")
    
    # 1. Compute delta relative to reference state
    delta = {name: update[name].detach() - reference[name].detach() for name in update}
    
    # 2. Compute total L2 norm across all parameters
    total_norm_sq = sum(torch.sum(val ** 2) for val in delta.values()).item()
    total_norm = float(total_norm_sq ** 0.5)
    
    # 3. Clip delta if norm > clip_norm
    scaling_factor = min(1.0, clip_norm / max(total_norm, 1e-8))
    clipped_delta = {name: val * scaling_factor for name, val in delta.items()}
    
    # 4. Add Gaussian noise N(0, (sigma * clip_norm)^2)
    torch.manual_seed(seed)
    noisy_delta = {}
    std_dev = noise_multiplier * clip_norm
    for name, val in clipped_delta.items():
        if std_dev > 0:
            noise = torch.randn_like(val) * std_dev
            noisy_delta[name] = val + noise
        else:
            noisy_delta[name] = val
            
    # 5. Reconstruct updated state
    return {name: reference[name].detach().clone() + noisy_delta[name] for name in reference}
