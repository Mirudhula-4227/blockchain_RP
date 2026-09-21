"""End-to-end test script for Python FabricClient integration with live network."""

from __future__ import annotations

import time
import uuid
from pathlib import Path

from src.crypto import generate_signing_key, sign_payload
from src.fabric_client import FabricClient
from src.ledger import SimulatedLedger


def main() -> None:
    print("==> Verifying Simulated Ledger & Tamper Detection Safety Net...")
    priv, pub = generate_signing_key()
    pld = {"round": 1, "update_hash": "hash123"}
    sig = sign_payload(priv, pld)
    tmp_path = Path("/tmp/test-tamper-ledger.json")
    if tmp_path.exists():
        tmp_path.unlink()
    sim_ledger = SimulatedLedger(tmp_path)
    entry = sim_ledger.append_update("client-0", 1, "hash123", pld, sig, pub)
    assert sim_ledger.verify(), "Valid simulated ledger failed verification!"

    # Tamper test
    sim_ledger.entries[0].update_hash = "tampered_hash"
    assert not sim_ledger.verify(), "Tampered ledger unexpectedly passed verification!"
    print("Simulated Ledger & Tamper Detection Passed!")

    client = FabricClient()
    print("==> Checking Fabric network availability...")
    available = client.is_available()
    print(f"Network available: {available}")
    assert available, "Fabric network is not available!"

    update_id = f"up-{uuid.uuid4().hex[:8]}"
    client_id = "client-py1"
    round_num = 1
    update_hash = "0xdeadbeef12345678"
    signature = "sig-py-ed25519-test"

    print(f"==> Submitting update {update_id}...")
    success = client.submit_update(update_id, client_id, round_num, update_hash, signature)
    assert success, "Failed to submit update to Fabric"

    # Allow block commitment
    time.sleep(2.0)

    print(f"==> Querying status for {update_id}...")
    status = client.get_status(update_id)
    print(f"Status: {status}")
    assert status["id"] == update_id
    assert status["status"] == "pending"

    print("==> Submitting votes...")
    client.vote(update_id, "val-1", True)
    time.sleep(2.0)

    client.vote(update_id, "val-2", True)
    time.sleep(2.0)

    status_after_votes = client.get_status(update_id)
    print(f"Status after votes: {status_after_votes}")
    assert status_after_votes["status"] == "approved"
    assert status_after_votes["votes"] == 2

    print("==> Setting client reputation score...")
    client.set_reputation(client_id, 0.985)
    time.sleep(2.0)

    rep = client.get_reputation(client_id)
    print(f"Reputation: {rep}")
    assert rep["clientId"] == client_id
    assert abs(rep["score"] - 0.985) < 1e-3

    print("\n✅ Live Fabric Network End-to-End Verification Passed Successfully!")


if __name__ == "__main__":
    main()
