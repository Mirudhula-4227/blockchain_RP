# Fabric handoff

The local safety net is complete in `src/crypto.py` and `src/ledger.py`. The Go chaincode contract is in `fabric/chaincode/go/chaincode.go` and exposes:

- `SubmitUpdate(id, clientId, round, hash, signature)`
- `Vote(updateId, validatorId, approve)`
- `Status(updateId)`
- `SetReputation(clientId, score)`
- `Reputation(clientId)`

This repository does not include a running Fabric network yet. Network startup requires the Fabric binaries, peer/orderer certificates, channel configuration, and Docker services. The chaincode should be packaged, installed, approved, and committed through the Fabric test-network or the team deployment environment after those prerequisites are available.

The contract deliberately stores update hashes and signatures, not model weights. Signature verification and cryptographic key custody remain client/validator responsibilities; Fabric endorsement and ordering provide the ledger transaction path.
