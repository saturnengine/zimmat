package linalg_test

import (
	"testing"

	"github.com/saturnengine/zimmat/linalg"
)

// testMatricesEqual checks if two matrices are equal (elements, rows, columns match within tolerance).
func testMatricesEqual(m1, m2 linalg.Matrix) bool {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return false
	}
	if len(m1.Data) != len(m2.Data) {
		return false
	}

	for i := range m1.Data {
		if !almostEqual(m1.Data[i], m2.Data[i]) {
			return false
		}
	}
	return true
}

// TestNewMatrix tests the NewMatrix function.
func TestNewMatrix(t *testing.T) {
	// Normal initialization (2x3)
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	m, err := linalg.NewMatrix(data)
	if err != nil {
		t.Fatalf("error occurred in NewMatrix: %v", err)
	}
	if m.Rows != 2 || m.Cols != 3 {
		t.Errorf("size differs from expected. expected: 2x3, actual: %dx%d", m.Rows, m.Cols)
	}
}

// TestMatrixGetAndSet tests the Get and Set methods.
func TestMatrixGetAndSet(t *testing.T) {
	data := [][]float64{{1.1, 2.2}, {3.3, 4.4}}
	m, _ := linalg.NewMatrix(data)

	// Test Get
	val, _ := m.Get(1, 0)
	if !almostEqual(val, 3.3) {
		t.Errorf("Get result differs from expected. expected: 3.3, actual: %f", val)
	}

	// Test Set
	m.Set(0, 1, 9.9)
	val, _ = m.Get(0, 1)
	if !almostEqual(val, 9.9) {
		t.Errorf("Set result is not reflected. expected: 9.9, actual: %f", val)
	}
}

// TestMatrixAdd tests the matrix addition method.
func TestMatrixAdd(t *testing.T) {
	m1, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	m2, _ := linalg.NewMatrix([][]float64{{5, 6}, {7, 8}})
	expected, _ := linalg.NewMatrix([][]float64{{6, 8}, {10, 12}})

	result, _ := m1.Add(m2)
	if !testMatricesEqual(result, expected) {
		t.Errorf("Add result differs from expected. expected: %v\nactual: %v", expected.Data, result.Data)
	}
}

// TestMatrixMultiply tests the matrix multiplication method.
func TestMatrixMultiply(t *testing.T) {
	// A (2x3) * B (3x2) = C (2x2)
	m_A, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}})
	m_B, _ := linalg.NewMatrix([][]float64{{7, 8}, {9, 10}, {11, 12}})
	expected, _ := linalg.NewMatrix([][]float64{{58, 64}, {139, 154}})

	result, _ := m_A.Multiply(m_B)
	if !testMatricesEqual(result, expected) {
		t.Errorf("Multiply result differs from expected.\nexpected: %v\nactual: %v", expected.Data, result.Data)
	}
}

// TestMatrixTranspose tests the matrix transpose method.
func TestMatrixTranspose(t *testing.T) {
	// 2x3 matrix
	m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}})
	// Expected 3x2 matrix
	expected, _ := linalg.NewMatrix([][]float64{{1, 4}, {2, 5}, {3, 6}})

	result := m.Transpose()

	if !testMatricesEqual(result, expected) {
		t.Errorf("Transpose result differs from expected.\nexpected: %v\nactual: %v", expected.Data, result.Data)
	}
	if result.Rows != 3 || result.Cols != 2 {
		t.Errorf("transposed size is incorrect. expected: 3x2, actual: %dx%d", result.Rows, result.Cols)
	}
}

// TestMatrixDeterminant tests the matrix determinant method.
func TestMatrixDeterminant(t *testing.T) {
	// 2x2: det(A) = 4*6 - 7*2 = 10
	m2x2, _ := linalg.NewMatrix([][]float64{{4, 7}, {2, 6}})
	det2x2, _ := m2x2.Determinant()
	if !almostEqual(det2x2, 10.0) {
		t.Errorf("2x2 determinant result is incorrect. expected: 10.0, actual: %f", det2x2)
	}

	// 3x3: det(A) = 27
	m3x3, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 0}})
	det3x3, _ := m3x3.Determinant()
	if !almostEqual(det3x3, 27.0) {
		t.Errorf("3x3 determinant result is incorrect. expected: 27.0, actual: %f", det3x3)
	}

	// Test non-square matrix
	mNonSquare, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}})
	_, err := mNonSquare.Determinant()
	if err == nil {
		t.Error("error was not returned for non-square matrix")
	}
}

// TestMatrixInverse tests the matrix inverse calculation method.
func TestMatrixInverse(t *testing.T) {
	// Normal 2x2 matrix (det=10)
	m2x2, _ := linalg.NewMatrix([][]float64{{4, 7}, {2, 6}})
	expected2x2, _ := linalg.NewMatrix([][]float64{{0.6, -0.7}, {-0.2, 0.4}})

	inv2x2, _ := m2x2.Inverse()
	if !testMatricesEqual(inv2x2, expected2x2) {
		t.Errorf("2x2 Inverse result differs from expected.\nexpected: %v\nactual: %v", expected2x2.Data, inv2x2.Data)
	}

	// Verification: check if A * A_inv equals identity matrix I
	product, _ := m2x2.Multiply(inv2x2)
	identity2x2, _ := linalg.NewMatrix([][]float64{{1.0, 0.0}, {0.0, 1.0}})
	if !testMatricesEqual(product, identity2x2) {
		t.Errorf("2x2 Inverse verification (A * A_inv) failed. result: %v", product.Data)
	}

	// Test singular matrix (inverse does not exist)
	singularM, _ := linalg.NewMatrix([][]float64{{2, 4}, {1, 2}})
	_, err := singularM.Inverse()
	if err == nil {
		t.Error("error was not returned for singular matrix")
	}
}

// TestMatrixSpecialTypes tests the diagonal, symmetric, and triangular matrix detection methods.
func TestMatrixSpecialTypes(t *testing.T) {
	// 1. Diagonal matrix
	diagM, _ := linalg.NewMatrix([][]float64{{1, 0}, {0, 2}})
	nonDiagM, _ := linalg.NewMatrix([][]float64{{1, 1}, {0, 2}})
	if !diagM.IsDiagonal() {
		t.Error("IsDiagonal: could not correctly identify diagonal matrix")
	}
	if nonDiagM.IsDiagonal() {
		t.Error("IsDiagonal: incorrectly identified non-diagonal matrix")
	}

	// 2. Symmetric matrix
	symM, _ := linalg.NewMatrix([][]float64{{1, 2}, {2, 3}})
	nonSymM, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	if !symM.IsSymmetric() {
		t.Error("IsSymmetric: could not correctly identify symmetric matrix")
	}
	if nonSymM.IsSymmetric() {
		t.Error("IsSymmetric: incorrectly identified non-symmetric matrix")
	}

	// 3. Upper triangular matrix
	upperM, _ := linalg.NewMatrix([][]float64{{1, 2}, {0, 3}})
	if !upperM.IsUpperTriangular() {
		t.Errorf("IsUpperTriangular: could not correctly identify upper triangular matrix")
	}
	if upperM.IsLowerTriangular() {
		t.Errorf("IsLowerTriangular: incorrectly identified upper triangular matrix")
	}

	// 4. Lower triangular matrix
	lowerM, _ := linalg.NewMatrix([][]float64{{1, 0}, {2, 3}})
	if !lowerM.IsLowerTriangular() {
		t.Errorf("IsLowerTriangular: could not correctly identify lower triangular matrix")
	}
	if lowerM.IsUpperTriangular() {
		t.Errorf("IsUpperTriangular: incorrectly identified lower triangular matrix")
	}
}

// TestNewMatrixFromTensor tests the NewMatrixFromTensor function.
func TestNewMatrixFromTensor(t *testing.T) {
	// Test successful conversion
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, _ := linalg.NewTensorWithData(data, 2, 3)
	m, err := linalg.NewMatrixFromTensor(tensor)
	if err != nil {
		t.Fatalf("error occurred in NewMatrixFromTensor: %v", err)
	}

	if m.Rows != 2 || m.Cols != 3 {
		t.Errorf("matrix size differs from expected. expected: 2x3, actual: %dx%d", m.Rows, m.Cols)
	}

	// Verify data
	expectedData := [][]float64{{1, 2, 3}, {4, 5, 6}}
	for i := 0; i < 2; i++ {
		for j := 0; j < 3; j++ {
			val, _ := m.Get(i, j)
			if !almostEqual(val, expectedData[i][j]) {
				t.Errorf("matrix element[%d,%d] differs from expected. expected: %f, actual: %f", i, j, expectedData[i][j], val)
			}
		}
	}

	// Test error case: non-matrix tensor
	tensor1d, _ := linalg.NewTensorWithData([]float64{1, 2, 3}, 3)
	_, err = linalg.NewMatrixFromTensor(tensor1d)
	if err == nil {
		t.Error("error was not returned for non-matrix tensor")
	}
}

// TestNewZeroMatrix tests the NewZeroMatrix function.
func TestNewZeroMatrix(t *testing.T) {
	m := linalg.NewZeroMatrix(3, 4)

	if m.Rows != 3 || m.Cols != 4 {
		t.Errorf("matrix size differs from expected. expected: 3x4, actual: %dx%d", m.Rows, m.Cols)
	}

	// Verify all elements are zero
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			val, _ := m.Get(i, j)
			if !almostEqual(val, 0.0) {
				t.Errorf("zero matrix element[%d,%d] should be 0.0, actual: %f", i, j, val)
			}
		}
	}
}

// TestNewIdentityMatrix tests the NewIdentityMatrix function.
func TestNewIdentityMatrix(t *testing.T) {
	m := linalg.NewIdentityMatrix(3)

	if m.Rows != 3 || m.Cols != 3 {
		t.Errorf("identity matrix size differs from expected. expected: 3x3, actual: %dx%d", m.Rows, m.Cols)
	}

	// Verify diagonal elements are 1 and off-diagonal elements are 0
	for i := 0; i < 3; i++ {
		for j := 0; j < 3; j++ {
			val, _ := m.Get(i, j)
			expected := 0.0
			if i == j {
				expected = 1.0
			}
			if !almostEqual(val, expected) {
				t.Errorf("identity matrix element[%d,%d] differs from expected. expected: %f, actual: %f", i, j, expected, val)
			}
		}
	}
}

// TestMatrixSubtract tests the matrix subtraction method.
func TestMatrixSubtract(t *testing.T) {
	m1, _ := linalg.NewMatrix([][]float64{{5, 6}, {7, 8}})
	m2, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	expected, _ := linalg.NewMatrix([][]float64{{4, 4}, {4, 4}})

	result, err := m1.Subtract(m2)
	if err != nil {
		t.Fatalf("error occurred in Subtract: %v", err)
	}

	if !testMatricesEqual(result, expected) {
		t.Errorf("Subtract result differs from expected. expected: %v\nactual: %v", expected.Data, result.Data)
	}

	// Test dimension mismatch
	m3, _ := linalg.NewMatrix([][]float64{{1, 2, 3}})
	_, err = m1.Subtract(m3)
	if err == nil {
		t.Error("error was not returned for dimension mismatch")
	}
}

// TestMatrixScale tests the matrix scaling method.
func TestMatrixScale(t *testing.T) {
	m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	expected, _ := linalg.NewMatrix([][]float64{{2.5, 5}, {7.5, 10}})

	result := m.Scale(2.5)

	if !testMatricesEqual(result, expected) {
		t.Errorf("Scale result differs from expected. expected: %v\nactual: %v", expected.Data, result.Data)
	}

	// Test scaling by zero
	zeroScaled := m.Scale(0)
	expectedZero, _ := linalg.NewMatrix([][]float64{{0, 0}, {0, 0}})

	if !testMatricesEqual(zeroScaled, expectedZero) {
		t.Errorf("Scale by zero result differs from expected. expected: %v\nactual: %v", expectedZero.Data, zeroScaled.Data)
	}
}

// TestMatrixClone tests the Clone method.
func TestMatrixClone(t *testing.T) {
	original, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	cloned := original.Clone()

	// Check if clone has same data
	if !testMatricesEqual(original, cloned) {
		t.Errorf("cloned matrix differs from original")
	}

	// Verify independence (modifying one should not affect the other)
	original.Set(0, 0, 99.0)
	clonedVal, _ := cloned.Get(0, 0)
	if almostEqual(clonedVal, 99.0) {
		t.Error("clone is not independent. original modification affects clone")
	}
}

// TestMatrixAsTensor tests the AsTensor method.
func TestMatrixAsTensor(t *testing.T) {
	m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	tensor := m.AsTensor()

	// Check tensor properties
	if tensor.Rank != 2 {
		t.Errorf("tensor rank differs from expected. expected: 2, actual: %d", tensor.Rank)
	}

	if len(tensor.Shape) != 2 || tensor.Shape[0] != 2 || tensor.Shape[1] != 2 {
		t.Errorf("tensor shape differs from expected. expected: [2, 2], actual: %v", tensor.Shape)
	}

	// Check data
	expected := []float64{1, 2, 3, 4}
	for i, expectedVal := range expected {
		if !almostEqual(tensor.Data[i], expectedVal) {
			t.Errorf("tensor data[%d] differs from expected. expected: %f, actual: %f", i, expectedVal, tensor.Data[i])
		}
	}

	// Verify independence
	tensor.Set(99.0, 0, 0)
	originalVal, _ := m.Get(0, 0)
	if almostEqual(originalVal, 99.0) {
		t.Error("tensor is not independent. tensor modification affects original matrix")
	}
}

// TestNewMatrixErrors tests error cases in matrix creation.
func TestNewMatrixErrors(t *testing.T) {
	// Test empty data
	_, err := linalg.NewMatrix([][]float64{})
	if err == nil {
		t.Error("error was not returned for empty data")
	}

	// Test inconsistent row lengths
	_, err = linalg.NewMatrix([][]float64{{1, 2}, {3, 4, 5}})
	if err == nil {
		t.Error("error was not returned for inconsistent row lengths")
	}
}

// TestMatrixGetSetErrors tests error cases in Get and Set methods.
func TestMatrixGetSetErrors(t *testing.T) {
	m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})

	// Test out-of-bounds Get
	_, err := m.Get(-1, 0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Get (negative row)")
	}

	_, err = m.Get(0, -1)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Get (negative column)")
	}

	_, err = m.Get(2, 0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Get (row too large)")
	}

	_, err = m.Get(0, 2)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Get (column too large)")
	}

	// Test out-of-bounds Set
	err = m.Set(-1, 0, 1.0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Set (negative row)")
	}

	err = m.Set(0, -1, 1.0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Set (negative column)")
	}

	err = m.Set(2, 0, 1.0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Set (row too large)")
	}

	err = m.Set(0, 2, 1.0)
	if err == nil {
		t.Error("error was not returned for out-of-bounds Set (column too large)")
	}
}

// TestMatrixMultiplyErrors tests error cases in matrix multiplication.
func TestMatrixMultiplyErrors(t *testing.T) {
	m1, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	m2, _ := linalg.NewMatrix([][]float64{{1, 2, 3}})

	_, err := m1.Multiply(m2)
	if err == nil {
		t.Error("error was not returned for incompatible matrix multiplication")
	}
}

// TestMatrixAddErrors tests error cases in matrix addition.
func TestMatrixAddErrors(t *testing.T) {
	m1, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	m2, _ := linalg.NewMatrix([][]float64{{1, 2, 3}})

	_, err := m1.Add(m2)
	if err == nil {
		t.Error("error was not returned for incompatible matrix addition")
	}
}

// TestMatrixSpecialTypesNonSquare tests special type checks on non-square matrices.
func TestMatrixSpecialTypesNonSquare(t *testing.T) {
	m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}})

	if m.IsDiagonal() {
		t.Error("IsDiagonal: non-square matrix should not be diagonal")
	}

	if m.IsSymmetric() {
		t.Error("IsSymmetric: non-square matrix should not be symmetric")
	}

	if m.IsUpperTriangular() {
		t.Error("IsUpperTriangular: non-square matrix should not be upper triangular")
	}

	if m.IsLowerTriangular() {
		t.Error("IsLowerTriangular: non-square matrix should not be lower triangular")
	}
}

// TestMatrixDeterminantExtended tests additional determinant cases.
func TestMatrixDeterminantExtended(t *testing.T) {
	// Test 1x1 matrix
	m1x1, _ := linalg.NewMatrix([][]float64{{5}})
	det1x1, _ := m1x1.Determinant()
	if !almostEqual(det1x1, 5.0) {
		t.Errorf("1x1 determinant result is incorrect. expected: 5.0, actual: %f", det1x1)
	}

	// Test singular 2x2 matrix (determinant = 0)
	singular2x2, _ := linalg.NewMatrix([][]float64{{1, 2}, {2, 4}})
	detSingular, _ := singular2x2.Determinant()
	if !almostEqual(detSingular, 0.0) {
		t.Errorf("singular 2x2 determinant result is incorrect. expected: 0.0, actual: %f", detSingular)
	}

	// Test 4x4 matrix (tests recursive cofactor expansion)
	m4x4, _ := linalg.NewMatrix([][]float64{
		{1, 0, 2, -1},
		{3, 0, 0, 5},
		{2, 1, 4, -3},
		{1, 0, 5, 0},
	})
	det4x4, _ := m4x4.Determinant()
	// Expected determinant calculated manually: 30
	if !almostEqual(det4x4, 30.0) {
		t.Errorf("4x4 determinant result is incorrect. expected: 30.0, actual: %f", det4x4)
	}
}
