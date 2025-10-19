package linalg_test

import (
	"math"
	"testing"

	"github.com/saturnengine/zimmat/linalg"
	"github.com/stretchr/testify/assert"
)

// Tolerance (epsilon) for float comparison
const floatTolerance = 1e-9

// almostEqual compares floating-point numbers within tolerance.
func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < floatTolerance
}

// testMatricesEqual checks if two matrices are equal (elements, rows, columns match within tolerance).
func testMatricesEqual(m1, m2 linalg.Matrix) bool {
	if m1.Rows() != m2.Rows() || m1.Cols() != m2.Cols() {
		return false
	}
	if len(m1.Data()) != len(m2.Data()) {
		return false
	}

	for i := range m1.Data() {
		if !almostEqual(m1.Data()[i], m2.Data()[i]) {
			return false
		}
	}
	return true
}

// TestNewMatrix tests the NewMatrix function with table-driven tests.
func TestNewMatrix(t *testing.T) {
	tests := []struct {
		name    string
		data    [][]float64
		wantErr bool
		rows    int
		cols    int
	}{
		{
			name: "valid 2x3 matrix",
			data: [][]float64{
				{1.0, 2.0, 3.0},
				{4.0, 5.0, 6.0},
			},
			wantErr: false,
			rows:    2,
			cols:    3,
		},
		{
			name:    "empty data",
			data:    [][]float64{},
			wantErr: true,
		},
		{
			name: "inconsistent row lengths",
			data: [][]float64{
				{1, 2},
				{3, 4, 5},
			},
			wantErr: true,
		},
		{
			name:    "single element matrix",
			data:    [][]float64{{42.0}},
			wantErr: false,
			rows:    1,
			cols:    1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := linalg.NewMatrix(tt.data)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.rows, m.Rows())
			assert.Equal(t, tt.cols, m.Cols())
		})
	}
}

// TestMatrixGetSet tests the Get and Set methods with table-driven tests.
func TestMatrixGetSet(t *testing.T) {
	m, _ := linalg.NewMatrix([][]float64{{1.1, 2.2}, {3.3, 4.4}})

	getTests := []struct {
		name     string
		row      int
		col      int
		expected float64
		wantErr  bool
	}{
		{"valid get (1,0)", 1, 0, 3.3, false},
		{"valid get (0,1)", 0, 1, 2.2, false},
		{"negative row", -1, 0, 0, true},
		{"negative col", 0, -1, 0, true},
		{"row too large", 2, 0, 0, true},
		{"col too large", 0, 2, 0, true},
	}

	for _, tt := range getTests {
		t.Run(tt.name, func(t *testing.T) {
			val, err := m.Get(tt.row, tt.col)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, almostEqual(val, tt.expected))
		})
	}

	setTests := []struct {
		name    string
		row     int
		col     int
		value   float64
		wantErr bool
	}{
		{"valid set", 0, 1, 9.9, false},
		{"negative row", -1, 0, 1.0, true},
		{"negative col", 0, -1, 1.0, true},
		{"row too large", 2, 0, 1.0, true},
		{"col too large", 0, 2, 1.0, true},
	}

	for _, tt := range setTests {
		t.Run(tt.name, func(t *testing.T) {
			err := m.Set(tt.row, tt.col, tt.value)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			val, _ := m.Get(tt.row, tt.col)
			assert.True(t, almostEqual(val, tt.value))
		})
	}
}

// TestMatrixAdd tests the matrix addition method with table-driven tests.
func TestMatrixAdd(t *testing.T) {
	tests := []struct {
		name     string
		m1       func() linalg.Matrix
		m2       func() linalg.Matrix
		expected func() linalg.Matrix
		wantErr  bool
	}{
		{
			name:     "valid 2x2 addition",
			m1:       func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			m2:       func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{5, 6}, {7, 8}}); return m },
			expected: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{6, 8}, {10, 12}}); return m },
			wantErr:  false,
		},
		{
			name:    "dimension mismatch",
			m1:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			m2:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}}); return m },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m1 := tt.m1()
			m2 := tt.m2()
			result, err := m1.Add(m2)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testMatricesEqual(result, tt.expected()))
		})
	}
}

// TestMatrixSubtract tests the matrix subtraction method with table-driven tests.
func TestMatrixSubtract(t *testing.T) {
	tests := []struct {
		name     string
		m1       func() linalg.Matrix
		m2       func() linalg.Matrix
		expected func() linalg.Matrix
		wantErr  bool
	}{
		{
			name:     "valid 2x2 subtraction",
			m1:       func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{5, 6}, {7, 8}}); return m },
			m2:       func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			expected: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{4, 4}, {4, 4}}); return m },
			wantErr:  false,
		},
		{
			name:    "dimension mismatch",
			m1:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{5, 6}, {7, 8}}); return m },
			m2:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}}); return m },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m1 := tt.m1()
			m2 := tt.m2()
			result, err := m1.Subtract(m2)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testMatricesEqual(result, tt.expected()))
		})
	}
}

// TestMatrixMultiply tests the matrix multiplication method with table-driven tests.
func TestMatrixMultiply(t *testing.T) {
	tests := []struct {
		name     string
		m1       func() linalg.Matrix
		m2       func() linalg.Matrix
		expected func() linalg.Matrix
		wantErr  bool
	}{
		{
			name:     "valid 2x3 * 3x2 multiplication",
			m1:       func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}}); return m },
			m2:       func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{7, 8}, {9, 10}, {11, 12}}); return m },
			expected: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{58, 64}, {139, 154}}); return m },
			wantErr:  false,
		},
		{
			name:    "incompatible dimensions",
			m1:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			m2:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}}); return m },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m1 := tt.m1()
			m2 := tt.m2()
			result, err := m1.Multiply(m2)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testMatricesEqual(result, tt.expected()))
		})
	}
}

// TestMatrixScale tests the matrix scaling method with table-driven tests.
func TestMatrixScale(t *testing.T) {
	tests := []struct {
		name     string
		matrix   func() linalg.Matrix
		scalar   float64
		expected func() linalg.Matrix
	}{
		{
			name:     "scale by 2.5",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			scalar:   2.5,
			expected: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{2.5, 5}, {7.5, 10}}); return m },
		},
		{
			name:     "scale by zero",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			scalar:   0,
			expected: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{0, 0}, {0, 0}}); return m },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.matrix()
			result := m.Scale(tt.scalar)
			assert.True(t, testMatricesEqual(result, tt.expected()))
		})
	}
}

// TestMatrixTranspose tests the matrix transpose method with table-driven tests.
func TestMatrixTranspose(t *testing.T) {
	tests := []struct {
		name      string
		matrix    func() linalg.Matrix
		expected  func() linalg.Matrix
		expectRow int
		expectCol int
	}{
		{
			name:      "2x3 transpose",
			matrix:    func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}}); return m },
			expected:  func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 4}, {2, 5}, {3, 6}}); return m },
			expectRow: 3,
			expectCol: 2,
		},
		{
			name:      "square matrix transpose",
			matrix:    func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			expected:  func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 3}, {2, 4}}); return m },
			expectRow: 2,
			expectCol: 2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.matrix()
			result := m.Transpose()
			assert.True(t, testMatricesEqual(result, tt.expected()))
			assert.Equal(t, tt.expectRow, result.Rows())
			assert.Equal(t, tt.expectCol, result.Cols())
		})
	}
}

// TestMatrixDeterminant tests the matrix determinant method with table-driven tests.
func TestMatrixDeterminant(t *testing.T) {
	tests := []struct {
		name     string
		matrix   func() linalg.Matrix
		expected float64
		wantErr  bool
	}{
		{
			name:     "2x2 determinant",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{4, 7}, {2, 6}}); return m },
			expected: 10.0,
			wantErr:  false,
		},
		{
			name:     "3x3 determinant",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 0}}); return m },
			expected: 27.0,
			wantErr:  false,
		},
		{
			name:     "1x1 determinant",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{5}}); return m },
			expected: 5.0,
			wantErr:  false,
		},
		{
			name:     "singular 2x2 determinant",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {2, 4}}); return m },
			expected: 0.0,
			wantErr:  false,
		},
		{
			name: "4x4 determinant",
			matrix: func() linalg.Matrix {
				m, _ := linalg.NewMatrix([][]float64{
					{1, 0, 2, -1},
					{3, 0, 0, 5},
					{2, 1, 4, -3},
					{1, 0, 5, 0},
				})
				return m
			},
			expected: 30.0,
			wantErr:  false,
		},
		{
			name:    "non-square matrix",
			matrix:  func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}}); return m },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.matrix()
			det, err := m.Determinant()
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, almostEqual(det, tt.expected))
		})
	}
}

// TestMatrixInverse tests the matrix inverse calculation method with table-driven tests.
func TestMatrixInverse(t *testing.T) {
	tests := []struct {
		name     string
		matrix   func() linalg.Matrix
		expected func() linalg.Matrix
		wantErr  bool
	}{
		{
			name:     "2x2 inverse",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{4, 7}, {2, 6}}); return m },
			expected: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{0.6, -0.7}, {-0.2, 0.4}}); return m },
			wantErr:  false,
		},
		{
			name:    "singular matrix",
			matrix:  func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{2, 4}, {1, 2}}); return m },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.matrix()
			inv, err := m.Inverse()
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testMatricesEqual(inv, tt.expected()))

			// Verification: check if A * A_inv equals identity matrix
			product, _ := m.Multiply(inv)
			identity, _ := linalg.NewMatrix([][]float64{{1.0, 0.0}, {0.0, 1.0}})
			assert.True(t, testMatricesEqual(product, identity))
		})
	}
}

// TestMatrixSpecialTypes tests the diagonal, symmetric, and triangular matrix detection methods.
func TestMatrixSpecialTypes(t *testing.T) {
	tests := []struct {
		name        string
		matrix      func() linalg.Matrix
		isDiagonal  bool
		isSymmetric bool
		isUpperTri  bool
		isLowerTri  bool
	}{
		{
			name:        "diagonal matrix",
			matrix:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 0}, {0, 2}}); return m },
			isDiagonal:  true,
			isSymmetric: true,
			isUpperTri:  true,
			isLowerTri:  true,
		},
		{
			name:        "symmetric matrix",
			matrix:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {2, 3}}); return m },
			isDiagonal:  false,
			isSymmetric: true,
			isUpperTri:  false,
			isLowerTri:  false,
		},
		{
			name:        "upper triangular matrix",
			matrix:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {0, 3}}); return m },
			isDiagonal:  false,
			isSymmetric: false,
			isUpperTri:  true,
			isLowerTri:  false,
		},
		{
			name:        "lower triangular matrix",
			matrix:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 0}, {2, 3}}); return m },
			isDiagonal:  false,
			isSymmetric: false,
			isUpperTri:  false,
			isLowerTri:  true,
		},
		{
			name:        "non-special matrix",
			matrix:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			isDiagonal:  false,
			isSymmetric: false,
			isUpperTri:  false,
			isLowerTri:  false,
		},
		{
			name:        "non-square matrix",
			matrix:      func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}}); return m },
			isDiagonal:  false,
			isSymmetric: false,
			isUpperTri:  false,
			isLowerTri:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.matrix()
			assert.Equal(t, tt.isDiagonal, m.IsDiagonal())
			assert.Equal(t, tt.isSymmetric, m.IsSymmetric())
			assert.Equal(t, tt.isUpperTri, m.IsUpperTriangular())
			assert.Equal(t, tt.isLowerTri, m.IsLowerTriangular())
		})
	}
}

// TestNewMatrixFromTensor tests the NewMatrixFromTensor function with table-driven tests.
func TestNewMatrixFromTensor(t *testing.T) {
	tests := []struct {
		name     string
		tensor   func() *linalg.Tensor
		expected func() linalg.Matrix
		wantErr  bool
	}{
		{
			name: "valid 2x3 tensor conversion",
			tensor: func() *linalg.Tensor {
				data := []float64{1, 2, 3, 4, 5, 6}
				tensor, _ := linalg.NewTensorWithData(data, 2, 3)
				return tensor
			},
			expected: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}}); return m },
			wantErr:  false,
		},
		{
			name: "non-matrix tensor (1D)",
			tensor: func() *linalg.Tensor {
				tensor, _ := linalg.NewTensorWithData([]float64{1, 2, 3}, 3)
				return tensor
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m, err := linalg.NewMatrixFromTensor(tt.tensor())
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testMatricesEqual(m, tt.expected()))
		})
	}
}

// TestNewZeroMatrix tests the NewZeroMatrix function with table-driven tests.
func TestNewZeroMatrix(t *testing.T) {
	tests := []struct {
		name string
		rows int
		cols int
	}{
		{"3x4 zero matrix", 3, 4},
		{"2x2 zero matrix", 2, 2},
		{"1x1 zero matrix", 1, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := linalg.NewZeroMatrix(tt.rows, tt.cols)
			assert.Equal(t, tt.rows, m.Rows())
			assert.Equal(t, tt.cols, m.Cols())

			// Verify all elements are zero
			for i := 0; i < tt.rows; i++ {
				for j := 0; j < tt.cols; j++ {
					val, _ := m.Get(i, j)
					assert.True(t, almostEqual(val, 0.0))
				}
			}
		})
	}
}

// TestNewIdentityMatrix tests the NewIdentityMatrix function with table-driven tests.
func TestNewIdentityMatrix(t *testing.T) {
	tests := []struct {
		name string
		size int
	}{
		{"3x3 identity", 3},
		{"2x2 identity", 2},
		{"1x1 identity", 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := linalg.NewIdentityMatrix(tt.size)
			assert.Equal(t, tt.size, m.Rows())
			assert.Equal(t, tt.size, m.Cols())

			// Verify diagonal elements are 1 and off-diagonal elements are 0
			for i := 0; i < tt.size; i++ {
				for j := 0; j < tt.size; j++ {
					val, _ := m.Get(i, j)
					expected := 0.0
					if i == j {
						expected = 1.0
					}
					assert.True(t, almostEqual(val, expected))
				}
			}
		})
	}
}

// TestMatrixClone tests the Clone method with table-driven tests.
func TestMatrixClone(t *testing.T) {
	tests := []struct {
		name   string
		matrix func() linalg.Matrix
	}{
		{
			name:   "2x2 matrix clone",
			matrix: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
		},
		{
			name:   "3x2 matrix clone",
			matrix: func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}, {5, 6}}); return m },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := tt.matrix()
			cloned := original.Clone()

			// Check if clone has same data
			assert.True(t, testMatricesEqual(original, cloned))

			// Verify independence
			original.Set(0, 0, 99.0)
			clonedVal, _ := cloned.Get(0, 0)
			assert.False(t, almostEqual(clonedVal, 99.0))
		})
	}
}

// TestMatrixAsTensor tests the AsTensor method with table-driven tests.
func TestMatrixAsTensor(t *testing.T) {
	tests := []struct {
		name     string
		matrix   func() linalg.Matrix
		expected []float64
	}{
		{
			name:     "2x2 matrix to tensor",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}}); return m },
			expected: []float64{1, 2, 3, 4},
		},
		{
			name:     "1x3 matrix to tensor",
			matrix:   func() linalg.Matrix { m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}}); return m },
			expected: []float64{1, 2, 3},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := tt.matrix()
			tensor := m.AsTensor()

			// Check tensor properties
			assert.Equal(t, 2, tensor.Rank())
			assert.Equal(t, []int{m.Rows(), m.Cols()}, tensor.Shape())

			// Check data
			for i, expectedVal := range tt.expected {
				assert.True(t, almostEqual(tensor.Data()[i], expectedVal))
			}

			// Verify independence
			tensor.Set(99.0, 0, 0)
			originalVal, _ := m.Get(0, 0)
			assert.False(t, almostEqual(originalVal, 99.0))
		})
	}
}
