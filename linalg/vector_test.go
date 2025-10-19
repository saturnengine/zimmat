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

// TestNewVectorFromTensor tests the NewVectorFromTensor function.
func TestNewVectorFromTensor(t *testing.T) {
	// Test successful conversion
	tensor, _ := linalg.NewTensorWithData([]float64{1, 2, 3}, 3)
	v, err := linalg.NewVectorFromTensor(tensor)
	if err != nil {
		t.Fatalf("error occurred in NewVectorFromTensor: %v", err)
	}

	if v.Dim != 3 {
		t.Errorf("vector dimension differs from expected. expected: 3, actual: %d", v.Dim)
	}

	expected := []float64{1, 2, 3}
	for i, val := range expected {
		if !almostEqual(v.Data[i], val) {
			t.Errorf("vector element[%d] differs from expected. expected: %f, actual: %f", i, val, v.Data[i])
		}
	}

	// Test error case: non-vector tensor
	tensor2d, _ := linalg.NewTensorWithData([]float64{1, 2, 3, 4}, 2, 2)
	_, err = linalg.NewVectorFromTensor(tensor2d)
	if err == nil {
		t.Error("error was not returned for non-vector tensor")
	}
}

// TestVectorSubtract tests the vector subtraction method.
func TestVectorSubtract(t *testing.T) {
	v1 := linalg.NewVector(5, 7, 9)
	v2 := linalg.NewVector(1, 2, 3)
	expected := linalg.NewVector(4, 5, 6)

	result, err := v1.Subtract(v2)
	if err != nil {
		t.Fatalf("error occurred in Subtract: %v", err)
	}

	if !testVectorsEqual(result, expected) {
		t.Errorf("Subtract result differs from expected. expected: %v, actual: %v", expected.Data, result.Data)
	}

	// Test dimension mismatch
	v3 := linalg.NewVector(1, 2)
	_, err = v1.Subtract(v3)
	if err == nil {
		t.Error("error was not returned for dimension mismatch")
	}
}

// TestVectorScale tests the vector scaling method.
func TestVectorScale(t *testing.T) {
	v := linalg.NewVector(1, 2, 3)
	expected := linalg.NewVector(2.5, 5.0, 7.5)

	result := v.Scale(2.5)

	if !testVectorsEqual(result, expected) {
		t.Errorf("Scale result differs from expected. expected: %v, actual: %v", expected.Data, result.Data)
	}

	// Test scaling by zero
	zeroScaled := v.Scale(0)
	expectedZero := linalg.NewVector(0, 0, 0)

	if !testVectorsEqual(zeroScaled, expectedZero) {
		t.Errorf("Scale by zero result differs from expected. expected: %v, actual: %v", expectedZero.Data, zeroScaled.Data)
	}
}

// TestVectorGetSet tests the Get and Set methods.
func TestVectorGetSet(t *testing.T) {
	v := linalg.NewVector(1, 2, 3)

	// Test Get
	val, err := v.Get(1)
	if err != nil {
		t.Fatalf("error occurred in Get: %v", err)
	}
	if !almostEqual(val, 2.0) {
		t.Errorf("Get result differs from expected. expected: 2.0, actual: %f", val)
	}

	// Test Set
	err = v.Set(1, 9.0)
	if err != nil {
		t.Fatalf("error occurred in Set: %v", err)
	}
	val, _ = v.Get(1)
	if !almostEqual(val, 9.0) {
		t.Errorf("Set result is not reflected. expected: 9.0, actual: %f", val)
	}

	// Test out-of-bounds Get
	_, err = v.Get(-1)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Get")
	}

	_, err = v.Get(3)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Get")
	}

	// Test out-of-bounds Set
	err = v.Set(-1, 1.0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Set")
	}

	err = v.Set(3, 1.0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Set")
	}
}

// TestVectorAsTensor tests the AsTensor method.
func TestVectorAsTensor(t *testing.T) {
	v := linalg.NewVector(1, 2, 3)
	tensor := v.AsTensor()

	// Check tensor properties
	if tensor.Rank != 1 {
		t.Errorf("tensor rank differs from expected. expected: 1, actual: %d", tensor.Rank)
	}

	if len(tensor.Shape) != 1 || tensor.Shape[0] != 3 {
		t.Errorf("tensor shape differs from expected. expected: [3], actual: %v", tensor.Shape)
	}

	// Check data
	for i, expected := range []float64{1, 2, 3} {
		val, _ := tensor.Get(i)
		if !almostEqual(val, expected) {
			t.Errorf("tensor element[%d] differs from expected. expected: %f, actual: %f", i, expected, val)
		}
	}

	// Verify independence (modifying tensor should not affect original vector)
	tensor.Set(99.0, 0)
	originalVal, _ := v.Get(0)
	if almostEqual(originalVal, 99.0) {
		t.Error("tensor is not independent. tensor modification affects original vector")
	}
}

// TestVectorClone tests the Clone method.
func TestVectorClone(t *testing.T) {
	original := linalg.NewVector(1, 2, 3)
	cloned := original.Clone()

	// Check if clone has same data
	if !testVectorsEqual(original, cloned) {
		t.Errorf("cloned vector differs from original. original: %v, cloned: %v", original.Data, cloned.Data)
	}

	// Verify independence (modifying one should not affect the other)
	original.Set(0, 99.0)
	clonedVal, _ := cloned.Get(0)
	if almostEqual(clonedVal, 99.0) {
		t.Error("clone is not independent. original modification affects clone")
	}
}

// TestVectorEmptyVector tests operations with empty vectors.
func TestVectorEmptyVector(t *testing.T) {
	empty := linalg.NewVector()

	if empty.Dim != 0 {
		t.Errorf("empty vector dimension should be 0. actual: %d", empty.Dim)
	}

	if len(empty.Data) != 0 {
		t.Errorf("empty vector data should be empty. actual length: %d", len(empty.Data))
	}

	// Test operations with empty vectors
	length := empty.Length()
	if length != 0 {
		t.Errorf("empty vector length should be 0. actual: %f", length)
	}

	lengthSq := empty.LengthSq()
	if lengthSq != 0 {
		t.Errorf("empty vector squared length should be 0. actual: %f", lengthSq)
	}
}

// TestVectorDotDimensionMismatch tests dot product with dimension mismatch.
func TestVectorDotDimensionMismatch(t *testing.T) {
	v1 := linalg.NewVector(1, 2, 3)
	v2 := linalg.NewVector(1, 2)

	_, err := v1.Dot(v2)
	if err == nil {
		t.Error("error was not returned for dimension mismatch in dot product")
	}
}
