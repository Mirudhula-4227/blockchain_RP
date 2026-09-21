"""Tamper-evident local ledger for signed federated updates."""

from __future__ import annotations

import base64
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from src.crypto import hash_payload, public_key_bytes, verify_signature


@dataclass
class LedgerEntry:
	index: int
	client_id: str
	round_number: int
	update_hash: str
	payload: dict[str, Any]
	signature: str
	public_key: str
	previous_hash: str
	entry_hash: str


class SimulatedLedger:
	def __init__(self, path: str | Path | None = None) -> None:
		self.path = Path(path) if path else None
		self.entries: list[LedgerEntry] = []
		if self.path and self.path.exists():
			self.entries = [LedgerEntry(**item) for item in json.loads(self.path.read_text(encoding="utf-8"))]

	def append_update(self, client_id: str, round_number: int, update_hash: str, payload: dict[str, Any], signature: str, public_key: Ed25519PublicKey) -> LedgerEntry:
		if not verify_signature(public_key, payload, signature):
			raise ValueError("invalid update signature")
		previous_hash = self.entries[-1].entry_hash if self.entries else "0" * 64
		entry_data = {"index": len(self.entries), "client_id": client_id, "round_number": round_number, "update_hash": update_hash, "payload": payload, "signature": signature, "public_key": public_key_bytes(public_key), "previous_hash": previous_hash}
		entry = LedgerEntry(**entry_data, entry_hash=hash_payload(entry_data))
		self.entries.append(entry)
		self._save()
		return entry

	def verify(self) -> bool:
		previous_hash = "0" * 64
		for expected_index, entry in enumerate(self.entries):
			if entry.index != expected_index or entry.previous_hash != previous_hash:
				return False
			entry_data = asdict(entry); entry_data.pop("entry_hash")
			if hash_payload(entry_data) != entry.entry_hash:
				return False
			try:
				public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(entry.public_key))
			except (ValueError, TypeError):
				return False
			if not verify_signature(public_key, entry.payload, entry.signature):
				return False
			previous_hash = entry.entry_hash
		return True

	def _save(self) -> None:
		if self.path:
			self.path.parent.mkdir(parents=True, exist_ok=True)
			self.path.write_text(json.dumps([asdict(entry) for entry in self.entries], indent=2), encoding="utf-8")
