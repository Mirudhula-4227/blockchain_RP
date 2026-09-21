package main

import (
	"encoding/json"
	"fmt"
	"time"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

type SmartContract struct { contractapi.Contract }

type Update struct {
	ID string `json:"id"`
	ClientID string `json:"clientId"`
	Round int `json:"round"`
	Hash string `json:"hash"`
	Signature string `json:"signature"`
	Status string `json:"status"`
	Votes int `json:"votes"`
	CreatedAt string `json:"createdAt"`
}

type Vote struct {
	UpdateID string `json:"updateId"`
	ValidatorID string `json:"validatorId"`
	Approve bool `json:"approve"`
}

type Reputation struct {
	ClientID string `json:"clientId"`
	Score float64 `json:"score"`
	UpdatedAt string `json:"updatedAt"`
}

func (s *SmartContract) SubmitUpdate(ctx contractapi.TransactionContextInterface, id, clientID string, round int, updateHash, signature string) error {
	if id == "" || clientID == "" || updateHash == "" || signature == "" { return fmt.Errorf("id, clientID, hash, and signature are required") }
	key, err := ctx.GetStub().CreateCompositeKey("update", []string{id}); if err != nil { return err }
	exists, err := ctx.GetStub().GetState(key); if err != nil { return err }; if exists != nil { return fmt.Errorf("update %s already exists", id) }
	update := Update{ID: id, ClientID: clientID, Round: round, Hash: updateHash, Signature: signature, Status: "pending", CreatedAt: time.Now().UTC().Format(time.RFC3339)}
	data, err := json.Marshal(update); if err != nil { return err }; return ctx.GetStub().PutState(key, data)
}

func (s *SmartContract) Vote(ctx contractapi.TransactionContextInterface, updateID, validatorID string, approve bool) error {
	key, err := ctx.GetStub().CreateCompositeKey("update", []string{updateID}); if err != nil { return err }
	data, err := ctx.GetStub().GetState(key); if err != nil || data == nil { return fmt.Errorf("update %s not found", updateID) }
	var update Update; if err := json.Unmarshal(data, &update); err != nil { return err }
	voteKey, err := ctx.GetStub().CreateCompositeKey("vote", []string{updateID, validatorID}); if err != nil { return err }
	if existing, _ := ctx.GetStub().GetState(voteKey); existing != nil { return fmt.Errorf("validator already voted") }
	voteData, err := json.Marshal(Vote{UpdateID: updateID, ValidatorID: validatorID, Approve: approve}); if err != nil { return err }
	if err := ctx.GetStub().PutState(voteKey, voteData); err != nil { return err }
	if approve { update.Votes++ }; if update.Votes >= 2 { update.Status = "approved" }
	updated, err := json.Marshal(update); if err != nil { return err }; return ctx.GetStub().PutState(key, updated)
}

func (s *SmartContract) Status(ctx contractapi.TransactionContextInterface, updateID string) (*Update, error) {
	key, err := ctx.GetStub().CreateCompositeKey("update", []string{updateID}); if err != nil { return nil, err }
	data, err := ctx.GetStub().GetState(key); if err != nil || data == nil { return nil, fmt.Errorf("update %s not found", updateID) }
	var update Update; return &update, json.Unmarshal(data, &update)
}

func (s *SmartContract) SetReputation(ctx contractapi.TransactionContextInterface, clientID string, score float64) error {
	if score < 0 || score > 1 { return fmt.Errorf("reputation score must be in [0,1]") }
	key, err := ctx.GetStub().CreateCompositeKey("reputation", []string{clientID}); if err != nil { return err }
	data, err := json.Marshal(Reputation{ClientID: clientID, Score: score, UpdatedAt: time.Now().UTC().Format(time.RFC3339)}); if err != nil { return err }
	return ctx.GetStub().PutState(key, data)
}

func (s *SmartContract) Reputation(ctx contractapi.TransactionContextInterface, clientID string) (*Reputation, error) {
	key, err := ctx.GetStub().CreateCompositeKey("reputation", []string{clientID}); if err != nil { return nil, err }
	data, err := ctx.GetStub().GetState(key); if err != nil || data == nil { return nil, fmt.Errorf("reputation for %s not found", clientID) }
	var reputation Reputation; return &reputation, json.Unmarshal(data, &reputation)
}

func main() { chaincode, err := contractapi.NewChaincode(&SmartContract{}); if err != nil { panic(err) }; if err := chaincode.Start(); err != nil { panic(err) } }
