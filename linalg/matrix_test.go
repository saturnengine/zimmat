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
