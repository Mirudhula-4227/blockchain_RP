#!/usr/bin/env bash
set -euo pipefail

# Deployment script for blockfed chaincode v2.0 on Fabric test-network
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="${HOME}/.local/bin:${ROOT_DIR}/fabric-samples/bin:${PATH}"
export FABRIC_CFG_PATH="${ROOT_DIR}/fabric-samples/config"

CHANNEL_NAME="blockfed"
CHAINCODE_NAME="blockfed"
VERSION="2.0"
SEQUENCE="2"
CC_SRC_PATH="${ROOT_DIR}/fabric/chaincode/go"
PKG_FILE="${ROOT_DIR}/fabric-samples/test-network/blockfed_2.tar.gz"

ORDERER_CA="${ROOT_DIR}/fabric-samples/test-network/organizations/ordererOrganizations/example.com/orderers/orderer.example.com/msp/tlscacerts/tlsca.example.com-cert.pem"
ORG1_CA="${ROOT_DIR}/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/tls/ca.crt"
ORG2_CA="${ROOT_DIR}/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/peers/peer0.org2.example.com/tls/ca.crt"

set_org1() {
    export CORE_PEER_TLS_ENABLED=true
    export CORE_PEER_LOCALMSPID="Org1MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE="${ORG1_CA}"
    export CORE_PEER_MSPCONFIGPATH="${ROOT_DIR}/fabric-samples/test-network/organizations/peerOrganizations/org1.example.com/users/Admin@org1.example.com/msp"
    export CORE_PEER_ADDRESS="localhost:7051"
}

set_org2() {
    export CORE_PEER_TLS_ENABLED=true
    export CORE_PEER_LOCALMSPID="Org2MSP"
    export CORE_PEER_TLS_ROOTCERT_FILE="${ORG2_CA}"
    export CORE_PEER_MSPCONFIGPATH="${ROOT_DIR}/fabric-samples/test-network/organizations/peerOrganizations/org2.example.com/users/Admin@org2.example.com/msp"
    export CORE_PEER_ADDRESS="localhost:9051"
}

echo "==> Packaging chaincode version ${VERSION} (sequence ${SEQUENCE})..."
peer lifecycle chaincode package "${PKG_FILE}" --path "${CC_SRC_PATH}" --lang golang --label "${CHAINCODE_NAME}_${VERSION}"

echo "==> Installing on Org1 peer..."
set_org1
peer lifecycle chaincode install "${PKG_FILE}"

echo "==> Installing on Org2 peer..."
set_org2
peer lifecycle chaincode install "${PKG_FILE}"

echo "==> Querying installed package ID..."
set_org1
PKG_ID=$(peer lifecycle chaincode queryinstalled | grep "${CHAINCODE_NAME}_${VERSION}" | sed -n 's/.*Package ID: \([^,]*\).*/\1/p')
echo "Package ID: ${PKG_ID}"

echo "==> Approving chaincode definition for Org1..."
set_org1
peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID "${CHANNEL_NAME}" --name "${CHAINCODE_NAME}" --version "${VERSION}" --package-id "${PKG_ID}" --sequence "${SEQUENCE}" --tls --cafile "${ORDERER_CA}"

echo "==> Approving chaincode definition for Org2..."
set_org2
peer lifecycle chaincode approveformyorg -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID "${CHANNEL_NAME}" --name "${CHAINCODE_NAME}" --version "${VERSION}" --package-id "${PKG_ID}" --sequence "${SEQUENCE}" --tls --cafile "${ORDERER_CA}"

echo "==> Committing chaincode definition to channel '${CHANNEL_NAME}'..."
set_org1
peer lifecycle chaincode commit -o localhost:7050 --ordererTLSHostnameOverride orderer.example.com --channelID "${CHANNEL_NAME}" --name "${CHAINCODE_NAME}" --version "${VERSION}" --sequence "${SEQUENCE}" --tls --cafile "${ORDERER_CA}" --peerAddresses localhost:7051 --tlsRootCertFiles "${ORG1_CA}" --peerAddresses localhost:9051 --tlsRootCertFiles "${ORG2_CA}"

echo "==> Querying committed chaincode on channel '${CHANNEL_NAME}'..."
peer lifecycle chaincode querycommitted --channelID "${CHANNEL_NAME}" --name "${CHAINCODE_NAME}"

echo "==> Chaincode deployment successful!"
