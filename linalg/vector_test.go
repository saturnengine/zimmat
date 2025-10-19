package linalg_test

import (
	"math"
	"testing"

	"github.com/saturnengine/zimmat/linalg"
)

// Tolerance (epsilon) for float comparison
const floatTolerance = 1e-9

// almostEqual compares floating-point numbers within tolerance.
func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < floatTolerance
}

// testVectorsEqual checks if two vectors are equal (elements and dimensions match within tolerance).
func testVectorsEqual(v1, v2 linalg.Vector) bool {
	if v1.Dim != v2.Dim {
		return false
	}
	for i := 0; i < v1.Dim; i++ {
		if !almostEqual(v1.Data[i], v2.Data[i]) {
			return false
		}
	}
	return true
}

// TestNewVector tests the NewVector function.
func TestNewVector(t *testing.T) {
	v := linalg.NewVector(1.0, 2.0, 3.0)
	if v.Dim != 3 {
		t.Errorf("dimension count differs from expected. expected: 3, actual: %d", v.Dim)
	}
	expected := []float64{1.0, 2.0, 3.0}
	for i := range expected {
		if !almostEqual(v.Data[i], expected[i]) {
			t.Errorf("element differs from expected. expected: %v, actual: %v", expected, v.Data)
			break
		}
	}
}

// TestVectorAdd tests the vector addition method.
func TestVectorAdd(t *testing.T) {
	v1 := linalg.NewVector(1, 2, 3)
	v2 := linalg.NewVector(4, 5, 6)
	expected := linalg.NewVector(5, 7, 9)

	result, err := v1.Add(v2)
	if err != nil {
		t.Fatalf("error occurred in Add: %v", err)
	}

	if !testVectorsEqual(result, expected) {
		t.Errorf("Add result differs from expected. expected: %v, actual: %v", expected.Data, result.Data)
	}

	// Test dimension mismatch
	v3 := linalg.NewVector(1, 2)
	_, err = v1.Add(v3)
	if err == nil {
		t.Error("error was not returned for dimension mismatch")
	}
}

// TestVectorDot tests the dot product method.
func TestVectorDot(t *testing.T) {
	v1 := linalg.NewVector(1, 0, 0)
	v2 := linalg.NewVector(0, 1, 0)
	v3 := linalg.NewVector(2, 3, 4)

	// Dot product of perpendicular vectors (0)
	dot1, _ := v1.Dot(v2)
	if !almostEqual(dot1, 0.0) {
		t.Errorf("dot product calculation is incorrect. expected: 0.0, actual: %f", dot1)
	}

	// General dot product (29)
	dot2, _ := v3.Dot(v3)
	if !almostEqual(dot2, 29.0) {
		t.Errorf("dot product calculation is incorrect. expected: 29.0, actual: %f", dot2)
	}
}

// TestVectorLengthAndNormalize tests the length and normalization methods.
func TestVectorLengthAndNormalize(t *testing.T) {
	v := linalg.NewVector(3, 4) // Vector of length 5

	// Test LengthSq
	if !almostEqual(v.LengthSq(), 25.0) {
		t.Errorf("LengthSq result is incorrect. expected: 25.0, actual: %f", v.LengthSq())
	}

	// Test Length
	if !almostEqual(v.Length(), 5.0) {
		t.Errorf("Length result is incorrect. expected: 5.0, actual: %f", v.Length())
	}

	// Test Normalize
	normalized, err := v.Normalize()
	if err != nil {
		t.Fatalf("error occurred in Normalize: %v", err)
	}
	expectedNormalized := linalg.NewVector(0.6, 0.8)

	if !testVectorsEqual(normalized, expectedNormalized) {
		t.Errorf("normalization result differs from expected. expected: %v, actual: %v", expectedNormalized.Data, normalized.Data)
	}

	// Verify that normalized vector has length 1
	if !almostEqual(normalized.Length(), 1.0) {
		t.Errorf("normalized vector length is not 1: %f", normalized.Length())
	}

	// Test zero vector normalization
	zeroV := linalg.NewVector(0, 0)
	_, err = zeroV.Normalize()
	if err == nil {
		t.Error("error was not returned for zero vector normalization")
	}
}
