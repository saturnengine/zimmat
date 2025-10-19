package zimmat_test

import (
	"testing"

	"github.com/saturnengine/zimmat"
)

// TestInit tests the Init function.
func TestInit(t *testing.T) {
	// Test that Init doesn't panic or error
	// Since Init prints to stdout, we can't easily capture the output in this simple test
	// but we can verify it doesn't crash
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("Init() panicked: %v", r)
		}
	}()

	zimmat.Init()
	// If we reach here without panic, the test passes
}

// TestVersion tests the Version constant.
func TestVersion(t *testing.T) {
	// Test that Version constant is accessible and has expected value
	if zimmat.Version == "" {
		t.Error("Version constant is empty")
	}

	expectedVersion := "0.1.0"
	if zimmat.Version != expectedVersion {
		t.Errorf("Version constant has unexpected value. expected: %s, actual: %s", expectedVersion, zimmat.Version)
	}
}

// TestPrecision tests the Precision constant.
func TestPrecision(t *testing.T) {
	// Test that Precision constant is accessible and has expected value
	expectedPrecision := 1e-9
	if zimmat.Precision != expectedPrecision {
		t.Errorf("Precision constant has unexpected value. expected: %e, actual: %e", expectedPrecision, zimmat.Precision)
	}

	// Precision should be positive
	if zimmat.Precision <= 0 {
		t.Error("Precision constant should be positive")
	}
}
