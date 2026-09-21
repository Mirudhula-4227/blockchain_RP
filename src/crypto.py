"""Canonical update hashing and Ed25519 signing helpers."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical_bytes(payload: Any) -> bytes:
	return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def hash_payload(payload: Any) -> str:
	return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def hash_state(state: Mapping[str, Any]) -> str:
	normalized = {}
	for name in sorted(state):
		value = state[name]
		if hasattr(value, "detach"):
			tensor = value.detach().cpu().contiguous()
			normalized[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape), "bytes": base64.b64encode(tensor.numpy().tobytes()).decode("ascii")}
		else:
			normalized[name] = value
	return hash_payload(normalized)


def generate_signing_key() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
	private_key = Ed25519PrivateKey.generate()
	return private_key, private_key.public_key()


def public_key_bytes(public_key: Ed25519PublicKey) -> str:
	return base64.b64encode(public_key.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode("ascii")


def sign_payload(private_key: Ed25519PrivateKey, payload: Any) -> str:
	return base64.b64encode(private_key.sign(canonical_bytes(payload))).decode("ascii")


def verify_signature(public_key: Ed25519PublicKey, payload: Any, signature: str) -> bool:
	try:
		public_key.verify(base64.b64decode(signature), canonical_bytes(payload))
	except (InvalidSignature, ValueError):
		return False
	return True
