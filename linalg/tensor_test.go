package linalg_test

import (
	"testing"

	"github.com/saturnengine/zimmat/linalg"
)

// TestNewTensor tests the NewTensor function.
func TestNewTensor(t *testing.T) {
	// Create 3-dimensional tensor (2x3x4)
	tensor := linalg.NewTensor(2, 3, 4)

	if tensor.Rank != 3 {
		t.Errorf("tensor rank differs from expected. expected: 3, actual: %d", tensor.Rank)
	}

	expectedShape := []int{2, 3, 4}
	for i, dim := range tensor.Shape {
		if dim != expectedShape[i] {
			t.Errorf("tensor shape[%d] differs from expected. expected: %d, actual: %d", i, expectedShape[i], dim)
		}
	}

	if tensor.Size() != 24 {
		t.Errorf("tensor size differs from expected. expected: 24, actual: %d", tensor.Size())
	}
}

// TestNewTensorWithData tests the NewTensorWithData function.
func TestNewTensorWithData(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, err := linalg.NewTensorWithData(data, 2, 3)

	if err != nil {
		t.Fatalf("error occurred in NewTensorWithData: %v", err)
	}

	if tensor.Rank != 2 {
		t.Errorf("tensor rank differs from expected. expected: 2, actual: %d", tensor.Rank)
	}

	val, _ := tensor.Get(1, 2)
	if !almostEqual(val, 6.0) {
		t.Errorf("tensor element[1,2] differs from expected. expected: 6.0, actual: %f", val)
	}
}

// TestTensorGetSet tests the Get/Set methods.
func TestTensorGetSet(t *testing.T) {
	tensor := linalg.NewTensor(3, 3)

	// Test Set
	err := tensor.Set(42.0, 1, 2)
	if err != nil {
		t.Fatalf("error occurred in Set operation: %v", err)
	}

	// Test Get
	val, err := tensor.Get(1, 2)
	if err != nil {
		t.Fatalf("error occurred in Get operation: %v", err)
	}

	if !almostEqual(val, 42.0) {
		t.Errorf("retrieved value differs from expected. expected: 42.0, actual: %f", val)
	}

	// Test out-of-bounds access
	_, err = tensor.Get(3, 0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds access")
	}
}

// TestTensorAdd tests the Add operation.
func TestTensorAdd(t *testing.T) {
	data1 := []float64{1, 2, 3, 4}
	data2 := []float64{5, 6, 7, 8}

	tensor1, _ := linalg.NewTensorWithData(data1, 2, 2)
	tensor2, _ := linalg.NewTensorWithData(data2, 2, 2)

	result, err := tensor1.Add(tensor2)
	if err != nil {
		t.Fatalf("error occurred in Add operation: %v", err)
	}

	expected := []float64{6, 8, 10, 12}
	for i, val := range result.Data {
		if !almostEqual(val, expected[i]) {
			t.Errorf("addition result[%d] differs from expected. expected: %f, actual: %f", i, expected[i], val)
		}
	}
}

// TestTensorSubtract tests the Subtract operation.
func TestTensorSubtract(t *testing.T) {
	data1 := []float64{5, 6, 7, 8}
	data2 := []float64{1, 2, 3, 4}

	tensor1, _ := linalg.NewTensorWithData(data1, 2, 2)
	tensor2, _ := linalg.NewTensorWithData(data2, 2, 2)

	result, err := tensor1.Subtract(tensor2)
	if err != nil {
		t.Fatalf("error occurred in Subtract operation: %v", err)
	}

	expected := []float64{4, 4, 4, 4}
	for i, val := range result.Data {
		if !almostEqual(val, expected[i]) {
			t.Errorf("subtraction result[%d] differs from expected. expected: %f, actual: %f", i, expected[i], val)
		}
	}
}

// TestTensorScale tests the Scale operation.
func TestTensorScale(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	tensor, _ := linalg.NewTensorWithData(data, 2, 2)

	result := tensor.Scale(2.0)

	expected := []float64{2, 4, 6, 8}
	for i, val := range result.Data {
		if !almostEqual(val, expected[i]) {
			t.Errorf("scale result[%d] differs from expected. expected: %f, actual: %f", i, expected[i], val)
		}
	}
}

// TestTensorReshape tests the Reshape operation.
func TestTensorReshape(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, _ := linalg.NewTensorWithData(data, 2, 3)

	reshaped, err := tensor.Reshape(3, 2)
	if err != nil {
		t.Fatalf("error occurred in Reshape operation: %v", err)
	}

	if reshaped.Rank != 2 {
		t.Errorf("reshaped rank differs from expected. expected: 2, actual: %d", reshaped.Rank)
	}

	expectedShape := []int{3, 2}
	for i, dim := range reshaped.Shape {
		if dim != expectedShape[i] {
			t.Errorf("reshaped shape[%d] differs from expected. expected: %d, actual: %d", i, expectedShape[i], dim)
		}
	}

	// Check if data is preserved
	val, _ := reshaped.Get(1, 1)
	if !almostEqual(val, 4.0) {
		t.Errorf("reshaped element[1,1] differs from expected. expected: 4.0, actual: %f", val)
	}
}

// TestTensorTranspose tests the transpose operation.
func TestTensorTranspose(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, _ := linalg.NewTensorWithData(data, 2, 3) // 2x3 matrix

	transposed, err := tensor.Transpose()
	if err != nil {
		t.Fatalf("error occurred in Transpose operation: %v", err)
	}

	if transposed.Rank != 2 {
		t.Errorf("transposed rank differs from expected. expected: 2, actual: %d", transposed.Rank)
	}

	expectedShape := []int{3, 2}
	for i, dim := range transposed.Shape {
		if dim != expectedShape[i] {
			t.Errorf("transposed shape[%d] differs from expected. expected: %d, actual: %d", i, expectedShape[i], dim)
		}
	}

	// Check if transpose is performed correctly
	val, _ := transposed.Get(1, 0)
	originalVal, _ := tensor.Get(0, 1)
	if !almostEqual(val, originalVal) {
		t.Errorf("transposed element[1,0] differs from expected. expected: %f, actual: %f", originalVal, val)
	}
}

// TestTensorMatrixMultiply tests matrix multiplication.
func TestTensorMatrixMultiply(t *testing.T) {
	// A (2x3) * B (3x2) = C (2x2)
	dataA := []float64{1, 2, 3, 4, 5, 6}
	dataB := []float64{7, 8, 9, 10, 11, 12}

	tensorA, _ := linalg.NewTensorWithData(dataA, 2, 3)
	tensorB, _ := linalg.NewTensorWithData(dataB, 3, 2)

	result, err := tensorA.MatrixMultiply(tensorB)
	if err != nil {
		t.Fatalf("error occurred in MatrixMultiply operation: %v", err)
	}

	// Expected result: [[58, 64], [139, 154]]
	expected := []float64{58, 64, 139, 154}
	for i, val := range result.Data {
		if !almostEqual(val, expected[i]) {
			t.Errorf("matrix multiplication result[%d] differs from expected. expected: %f, actual: %f", i, expected[i], val)
		}
	}
}

// TestTensorVectorDot tests vector dot product.
func TestTensorVectorDot(t *testing.T) {
	data1 := []float64{1, 2, 3}
	data2 := []float64{4, 5, 6}

	tensor1, _ := linalg.NewTensorWithData(data1, 3)
	tensor2, _ := linalg.NewTensorWithData(data2, 3)

	dot, err := tensor1.VectorDot(tensor2)
	if err != nil {
		t.Fatalf("error occurred in VectorDot operation: %v", err)
	}

	expected := 32.0 // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	if !almostEqual(dot, expected) {
		t.Errorf("dot product result differs from expected. expected: %f, actual: %f", expected, dot)
	}
}

// TestTensorVectorLength tests vector length calculation.
func TestTensorVectorLength(t *testing.T) {
	data := []float64{3, 4} // Vector of length 5
	tensor, _ := linalg.NewTensorWithData(data, 2)

	length, err := tensor.VectorLength()
	if err != nil {
		t.Fatalf("error occurred in VectorLength operation: %v", err)
	}

	expected := 5.0
	if !almostEqual(length, expected) {
		t.Errorf("vector length differs from expected. expected: %f, actual: %f", expected, length)
	}
}

// TestTensorVectorNormalize tests vector normalization.
func TestTensorVectorNormalize(t *testing.T) {
	data := []float64{3, 4} // Vector of length 5
	tensor, _ := linalg.NewTensorWithData(data, 2)

	normalized, err := tensor.VectorNormalize()
	if err != nil {
		t.Fatalf("error occurred in VectorNormalize operation: %v", err)
	}

	// Verify that normalized vector has length 1
	length, _ := normalized.VectorLength()
	if !almostEqual(length, 1.0) {
		t.Errorf("normalized vector length is not 1: %f", length)
	}

	// Check if normalized vector elements are correct
	val0, _ := normalized.Get(0)
	val1, _ := normalized.Get(1)
	if !almostEqual(val0, 0.6) || !almostEqual(val1, 0.8) {
		t.Errorf("normalized vector elements differ from expected. expected: [0.6, 0.8], actual: [%f, %f]", val0, val1)
	}
}

// TestTensorAsVectorMatrix tests the AsVector/AsMatrix methods.
func TestTensorAsVectorMatrix(t *testing.T) {
	// Convert 1-dimensional tensor to vector
	data1d := []float64{1, 2, 3}
	tensor1d, _ := linalg.NewTensorWithData(data1d, 3)

	vector, err := tensor1d.AsVector()
	if err != nil {
		t.Fatalf("error occurred in AsVector conversion: %v", err)
	}

	if vector.Dim != 3 {
		t.Errorf("converted vector dimension differs from expected. expected: 3, actual: %d", vector.Dim)
	}

	// Convert 2-dimensional tensor to matrix
	data2d := []float64{1, 2, 3, 4}
	tensor2d, _ := linalg.NewTensorWithData(data2d, 2, 2)

	matrix, err := tensor2d.AsMatrix()
	if err != nil {
		t.Fatalf("error occurred in AsMatrix conversion: %v", err)
	}

	if matrix.Rows != 2 || matrix.Cols != 2 {
		t.Errorf("converted matrix size differs from expected. expected: 2x2, actual: %dx%d", matrix.Rows, matrix.Cols)
	}
}

// TestTensorClone tests the Clone method.
func TestTensorClone(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	original, _ := linalg.NewTensorWithData(data, 2, 2)

	cloned := original.Clone()

	// Check if clone has same data
	for i := range original.Data {
		if !almostEqual(original.Data[i], cloned.Data[i]) {
			t.Errorf("cloned data[%d] differs. expected: %f, actual: %f", i, original.Data[i], cloned.Data[i])
		}
	}

	// Verify independence (modifying one should not affect the other)
	original.Set(99.0, 0, 0)
	clonedVal, _ := cloned.Get(0, 0)
	if almostEqual(clonedVal, 99.0) {
		t.Error("clone is not independent. original modification affects clone")
	}
}
