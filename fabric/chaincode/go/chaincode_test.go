package main

import (
	"testing"
)

func TestSmartContractStruct(t *testing.T) {
	contract := new(SmartContract)
	if contract == nil {
		t.Fatalf("failed to instantiate SmartContract")
	}
}
