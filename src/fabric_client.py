"""Hyperledger Fabric client wrapper using peer CLI for BlockFed-IDS smart contract."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


class FabricClient:
    def __init__(self, root_dir: str | Path | None = None, channel: str = "blockfed", chaincode: str = "blockfed") -> None:
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parent.parent
        self.channel = channel
        self.chaincode = chaincode

        fabric_samples_dir = self.root_dir / "fabric-samples"
        if not fabric_samples_dir.exists():
            fallback_dir = Path("/Users/mirudhulam/blockfed-ids/fabric-samples")
            if fallback_dir.exists():
                fabric_samples_dir = fallback_dir

        local_bin = Path.home() / ".local" / "bin"
        fabric_bin = fabric_samples_dir / "bin"
        current_path = os.environ.get("PATH", "")
        self.env = os.environ.copy()
        self.env["PATH"] = f"{local_bin}:{fabric_bin}:{current_path}"
        self.env["FABRIC_CFG_PATH"] = str(fabric_samples_dir / "config")
        self.env["CORE_PEER_TLS_ENABLED"] = "true"
        self.env["CORE_PEER_LOCALMSPID"] = "Org1MSP"
        self.env["CORE_PEER_TLS_ROOTCERT_FILE"] = str(
            fabric_samples_dir / "test-network" / "organizations" / "peerOrganizations" / "org1.example.com" / "peers" / "peer0.org1.example.com" / "tls" / "ca.crt"
        )
        self.env["CORE_PEER_MSPCONFIGPATH"] = str(
            fabric_samples_dir / "test-network" / "organizations" / "peerOrganizations" / "org1.example.com" / "users" / "Admin@org1.example.com" / "msp"
        )
        self.env["CORE_PEER_ADDRESS"] = "localhost:7051"

        self.orderer_ca = str(
            fabric_samples_dir / "test-network" / "organizations" / "ordererOrganizations" / "example.com" / "orderers" / "orderer.example.com" / "msp" / "tlscacerts" / "tlsca.example.com-cert.pem"
        )
        self.org1_ca = str(
            fabric_samples_dir / "test-network" / "organizations" / "peerOrganizations" / "org1.example.com" / "peers" / "peer0.org1.example.com" / "tls" / "ca.crt"
        )
        self.org2_ca = str(
            fabric_samples_dir / "test-network" / "organizations" / "peerOrganizations" / "org2.example.com" / "peers" / "peer0.org2.example.com" / "tls" / "ca.crt"
        )

    def is_available(self) -> bool:
        cmd = [
            "peer", "lifecycle", "chaincode", "querycommitted",
            "--channelID", self.channel,
            "--name", self.chaincode,
        ]
        try:
            res = subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=10)
            return res.returncode == 0 and self.chaincode in res.stdout
        except Exception:
            return False

    def _query(self, function_name: str, *args: str) -> dict[str, Any]:
        payload = json.dumps({"Args": [function_name, *args]})
        cmd = [
            "peer", "chaincode", "query",
            "-C", self.channel,
            "-n", self.chaincode,
            "-c", payload,
        ]
        res = subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=10)
        if res.returncode != 0:
            raise RuntimeError(f"Fabric query {function_name} failed: {res.stderr.strip()}")
        return json.loads(res.stdout.strip())

    def _invoke(self, function_name: str, *args: str, max_retries: int = 3) -> bool:
        payload = json.dumps({"Args": [function_name, *args]})
        cmd = [
            "peer", "chaincode", "invoke",
            "-o", "localhost:7050",
            "--ordererTLSHostnameOverride", "orderer.example.com",
            "--tls",
            "--cafile", self.orderer_ca,
            "-C", self.channel,
            "-n", self.chaincode,
            "--peerAddresses", "localhost:7051",
            "--tlsRootCertFiles", self.org1_ca,
            "--peerAddresses", "localhost:9051",
            "--tlsRootCertFiles", self.org2_ca,
            "-c", payload,
        ]

        for attempt in range(max_retries):
            res = subprocess.run(cmd, env=self.env, capture_output=True, text=True, timeout=15)
            combined_output = res.stdout + "\n" + res.stderr
            if res.returncode == 0 and ("status:200" in combined_output or "Chaincode invoke successful" in combined_output):
                return True
            time.sleep(1.0)
        raise RuntimeError(f"Fabric invoke {function_name} failed after {max_retries} attempts:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}")

    def submit_update(self, update_id: str, client_id: str, round_num: int, update_hash: str, signature: str) -> bool:
        return self._invoke("SubmitUpdate", update_id, client_id, str(round_num), update_hash, signature)

    def vote(self, update_id: str, validator_id: str, approve: bool) -> bool:
        return self._invoke("Vote", update_id, validator_id, "true" if approve else "false")

    def get_status(self, update_id: str) -> dict[str, Any]:
        return self._query("Status", update_id)

    def set_reputation(self, client_id: str, score: float) -> bool:
        return self._invoke("SetReputation", client_id, f"{score:.4f}")

    def get_reputation(self, client_id: str) -> dict[str, Any]:
        return self._query("Reputation", client_id)
