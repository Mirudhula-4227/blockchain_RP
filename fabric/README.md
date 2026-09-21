# Fabric Architecture & Live Deployment

The local safety net is complete in `src/crypto.py` and `src/ledger.py`. The Go chaincode contract in `fabric/chaincode/go/chaincode.go` utilizes deterministic timestamps via `ctx.GetStub().GetTxTimestamp()` and exposes:

- `SubmitUpdate(id, clientId, round, hash, signature)`
- `Vote(updateId, validatorId, approve)`
- `Status(updateId)`
- `SetReputation(clientId, score)`
- `Reputation(clientId)`

## Live Fabric Network

The Hyperledger Fabric network runs via Docker on channel `blockfed` (peer0.org1:7051, peer0.org2:9051, orderer:7050). Go 1.24+ and Fabric binaries (`peer`, `orderer`, `configtxgen`, `cryptogen`) are configured.

- Chaincode Deployment Script: `fabric/deploy_chaincode.sh`
- Python Fabric Client: `src/fabric_client.py`
- End-to-End Verification: `python -m src.verify_fabric`
- Federated Learning Integration: `python -m src.fl --ledger fabric`

The contract stores update payload hashes and Ed25519 signatures, not raw model weights. Signature verification and key custody are handled by client nodes; Fabric endorsement and ordering provide tamper-resistant consensus.

