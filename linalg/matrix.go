package linalg

import (
	"fmt"
	"math"
)

// Matrix represents an n×m matrix optimized for linear algebra operations.
// It is implemented internally using a Tensor for consistency and performance.
//
// Matrix provides comprehensive functionality for matrix operations including
// arithmetic, multiplication, transposition, determinant calculation, and inversion.
//
// Fields:
//   - tensor: Internal 2-dimensional tensor
//   - Data: Public field for compatibility (reference to tensor's Data)
//   - Rows: Number of rows (n)
//   - Cols: Number of columns (m)
//
// Example:
//
//	data := [][]float64{{1, 2}, {3, 4}}
//	m, err := linalg.NewMatrix(data)
//	if err != nil {
//		panic(err)
//	}
//	det, _ := m.Determinant() // Returns -2.0
type Matrix struct {
	tensor *Tensor   // Internal tensor (2-dimensional)
	Data   []float64 // Public field for compatibility (reference to tensor's Data)
	Rows   int       // Number of rows (n)
	Cols   int       // Number of columns (m)
}

// NewMatrix creates a new Matrix from a 2D slice of float64 values.
// Each inner slice represents a row of the matrix.
//
// Parameters:
//   - data: 2D slice where data[i][j] represents the element at row i, column j
//
// Returns:
//   - Matrix: The new matrix
//   - error: Non-nil if data is empty or has inconsistent row lengths
//
// All rows must have the same number of columns.
//
// Example:
//
//	data := [][]float64{
//		{1.0, 2.0, 3.0},
//		{4.0, 5.0, 6.0},
//	}
//	m, err := linalg.NewMatrix(data) // Creates 2x3 matrix
func NewMatrix(data [][]float64) (result Matrix, err error) {
	if len(data) == 0 {
		err = fmt.Errorf("input data has no rows")
		return
	}

	rows := len(data)
	cols := len(data[0])

	flatData := make([]float64, 0, rows*cols)

	for i, rowData := range data {
		if len(rowData) != cols {
			err = fmt.Errorf("row %d has different column count: expected %d, actual %d", i, cols, len(rowData))
			return
		}
		flatData = append(flatData, rowData...)
	}

	tensor, err := NewTensorWithData(flatData, rows, cols)
	if err != nil {
		err = fmt.Errorf("matrix creation error: %v", err)
		return
	}

	result = Matrix{
		tensor: tensor,
		Data:   tensor.Data,
		Rows:   rows,
		Cols:   cols,
	}
	return
}

// NewMatrixFromTensor creates a Matrix from an existing Tensor.
// The tensor must be 2-dimensional (matrix).
//
// Parameters:
//   - tensor: A 2-dimensional tensor to convert
//
// Returns:
//   - Matrix: The new matrix
//   - error: Non-nil if the tensor is not 2-dimensional
//
// Example:
//
//	t := linalg.NewTensor(2, 3) // 2x3 tensor
//	m, err := linalg.NewMatrixFromTensor(t)
//	if err != nil {
//		panic(err)
//	}
func NewMatrixFromTensor(tensor *Tensor) (result Matrix, err error) {
	if !tensor.IsMatrix() {
		err = fmt.Errorf("tensor with shape %v is not a matrix", tensor.Shape)
		return
	}

	result = Matrix{
		tensor: tensor.Clone(),
		Data:   tensor.Data,
		Rows:   tensor.Shape[0],
		Cols:   tensor.Shape[1],
	}
	return
}

// NewZeroMatrix creates a zero matrix (all elements are zero) with the specified dimensions.
//
// Parameters:
//   - rows: Number of rows
//   - cols: Number of columns
//
// Returns:
//   - Matrix: A new zero matrix
//
// Example:
//
//	zeros := linalg.NewZeroMatrix(3, 4) // 3x4 matrix of zeros
func NewZeroMatrix(rows, cols int) (result Matrix) {
	tensor := NewTensor(rows, cols)
	result = Matrix{
		tensor: tensor,
		Data:   tensor.Data,
		Rows:   rows,
		Cols:   cols,
	}
	return
}

// NewIdentityMatrix creates an identity matrix (1s on diagonal, 0s elsewhere) of the specified size.
// Identity matrices are always square (n×n).
//
// Parameters:
//   - size: The size of the square matrix (both rows and columns)
//
// Returns:
//   - Matrix: A new identity matrix
//
// Example:
//
//	identity := linalg.NewIdentityMatrix(3) // 3x3 identity matrix
func NewIdentityMatrix(size int) (result Matrix) {
	result = NewZeroMatrix(size, size)
	for i := 0; i < size; i++ {
		result.Set(i, i, 1.0)
	}
	return
}

// Get retrieves the element at the specified row and column.
//
// Parameters:
//   - row: Row index (0-based)
//   - col: Column index (0-based)
//
// Returns:
//   - float64: The value at the specified position
//   - error: Non-nil if indices are out of bounds
//
// Example:
//
//	data := [][]float64{{1, 2}, {3, 4}}
//	m, _ := linalg.NewMatrix(data)
//	val, err := m.Get(1, 0) // Returns 3.0
func (m Matrix) Get(row, col int) (value float64, err error) {
	value, err = m.tensor.Get(row, col)
	return
}

// Set assigns a value to the element at the specified row and column.
//
// Parameters:
//   - row: Row index (0-based)
//   - col: Column index (0-based)
//   - val: The value to assign
//
// Returns:
//   - error: Non-nil if indices are out of bounds
//
// Example:
//
//	m := linalg.NewZeroMatrix(2, 2)
//	err := m.Set(0, 1, 5.0) // Sets element at row 0, column 1 to 5.0
func (m Matrix) Set(row, col int, val float64) (err error) {
	err = m.tensor.Set(val, row, col)
	if err != nil {
		return
	}
	// Update Data field (it's automatically updated since it's a reference, but explicitly ensure it)
	m.Data = m.tensor.Data
	return
}

// Add returns a new matrix that is the element-wise sum of this matrix and another matrix.
func (m Matrix) Add(other Matrix) (matrixResult Matrix, err error) {
	result, err := m.tensor.Add(other.tensor)
	if err != nil {
		err = fmt.Errorf("matrix addition error: %v", err)
		return
	}

	matrixResult, err = NewMatrixFromTensor(result)
	if err != nil {
		err = fmt.Errorf("result matrix creation error: %v", err)
		return
	}

	return
}

// Subtract returns a new matrix that is the element-wise difference of this matrix and another matrix.
func (m Matrix) Subtract(other Matrix) (matrixResult Matrix, err error) {
	result, err := m.tensor.Subtract(other.tensor)
	if err != nil {
		err = fmt.Errorf("matrix subtraction error: %v", err)
		return
	}

	matrixResult, err = NewMatrixFromTensor(result)
	if err != nil {
		err = fmt.Errorf("result matrix creation error: %v", err)
		return
	}

	return
}

// Scale returns a new matrix that is this matrix multiplied by a scalar value.
func (m Matrix) Scale(scalar float64) (matrixResult Matrix) {
	result := m.tensor.Scale(scalar)

	matrixResult, err := NewMatrixFromTensor(result)
	if err != nil {
		// Return zero matrix instead of panicking
		matrixResult = NewZeroMatrix(1, 1)
		return
	}

	return
}

// Multiply returns a new matrix that is the product of this matrix (A) and another matrix (B), resulting in C = A * B.
func (m Matrix) Multiply(other Matrix) (matrixResult Matrix, err error) {
	result, err := m.tensor.MatrixMultiply(other.tensor)
	if err != nil {
		err = fmt.Errorf("matrix multiplication error: %v", err)
		return
	}

	matrixResult, err = NewMatrixFromTensor(result)
	if err != nil {
		err = fmt.Errorf("result matrix creation error: %v", err)
		return
	}

	return
}

// Transpose returns the transpose of the current matrix.
func (m Matrix) Transpose() (matrixResult Matrix) {
	result, err := m.tensor.Transpose()
	if err != nil {
		// Return zero matrix instead of panicking
		matrixResult = NewZeroMatrix(1, 1)
		return
	}

	matrixResult, err = NewMatrixFromTensor(result)
	if err != nil {
		// Return zero matrix instead of panicking
		matrixResult = NewZeroMatrix(1, 1)
		return
	}

	return
}

// AsTensor returns a copy of this matrix as a Tensor.
func (m Matrix) AsTensor() (result *Tensor) {
	result = m.tensor.Clone()
	return
}

// Clone creates a complete deep copy of the matrix.
func (m Matrix) Clone() (result Matrix) {
	clonedTensor := m.tensor.Clone()
	result = Matrix{
		tensor: clonedTensor,
		Data:   clonedTensor.Data,
		Rows:   m.Rows,
		Cols:   m.Cols,
	}
	return
}

// Determinant calculates the determinant of a square matrix.
// For matrices up to 3x3, it uses direct formulas/cofactor expansion.
// For larger matrices, it uses recursive cofactor expansion (computationally expensive).
func (m Matrix) Determinant() (det float64, err error) {
	if m.Rows != m.Cols {
		err = fmt.Errorf("determinant is only defined for square matrices (%d x %d)", m.Rows, m.Cols)
		return
	}
	n := m.Rows

	if n == 1 {
		det, _ = m.Get(0, 0)
		return
	}
	if n == 2 {
		// ad - bc
		a, _ := m.Get(0, 0)
		b, _ := m.Get(0, 1)
		c, _ := m.Get(1, 0)
		d, _ := m.Get(1, 1)
		det = a*d - b*c
		return
	}

	// For 3x3 and larger, use recursive cofactor expansion (high computational cost)
	for j := 0; j < n; j++ {
		// Cofactor expansion along row i=0
		sign := 1.0
		if j%2 != 0 {
			sign = -1.0
		}

		// Create minor matrix
		minorData := make([][]float64, n-1)
		for row := 1; row < n; row++ { // Exclude row 0
			minorRow := make([]float64, 0, n-1)
			for col := 0; col < n; col++ {
				if col != j { // Exclude column j
					val, _ := m.Get(row, col)
					minorRow = append(minorRow, val)
				}
			}
			minorData[row-1] = minorRow
		}

		minorM, _ := NewMatrix(minorData) // Minor matrix
		minorDet, _ := minorM.Determinant()

		a0j, _ := m.Get(0, j)
		det += sign * a0j * minorDet
	}

	return
}

// Inverse calculates the inverse matrix of the current matrix using Gauss-Jordan elimination.
func (m Matrix) Inverse() (result Matrix, err error) {
	if m.Rows != m.Cols {
		err = fmt.Errorf("inverse is only defined for square matrices (%d x %d)", m.Rows, m.Cols)
		return
	}

	n := m.Rows
	// Create augmented matrix [A|I]
	augmented := NewZeroMatrix(n, 2*n)

	// Copy original matrix to left side
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			val, _ := m.Get(i, j)
			augmented.Set(i, j, val)
		}
	}

	// Set identity matrix on right side
	for i := 0; i < n; i++ {
		augmented.Set(i, n+i, 1.0)
	}

	const epsilon = 1e-9

	// Gauss-Jordan elimination
	for i := 0; i < n; i++ {
		// Pivot selection
		pivotRow := i
		for k := i + 1; k < n; k++ {
			currentVal, _ := augmented.Get(k, i)
			pivotVal, _ := augmented.Get(pivotRow, i)
			if math.Abs(currentVal) > math.Abs(pivotVal) {
				pivotRow = k
			}
		}

		pivotVal, _ := augmented.Get(pivotRow, i)
		if math.Abs(pivotVal) < epsilon {
			err = fmt.Errorf("matrix is singular (determinant close to zero): inverse does not exist")
			return
		}

		// Row swap
		if pivotRow != i {
			for j := 0; j < 2*n; j++ {
				val1, _ := augmented.Get(i, j)
				val2, _ := augmented.Get(pivotRow, j)
				augmented.Set(i, j, val2)
				augmented.Set(pivotRow, j, val1)
			}
		}

		// Normalize pivot row
		pivotVal, _ = augmented.Get(i, i)
		for j := i; j < 2*n; j++ {
			val, _ := augmented.Get(i, j)
			augmented.Set(i, j, val/pivotVal)
		}

		// Eliminate other rows
		for k := 0; k < n; k++ {
			if k != i {
				factor, _ := augmented.Get(k, i)
				for j := i; j < 2*n; j++ {
					kVal, _ := augmented.Get(k, j)
					iVal, _ := augmented.Get(i, j)
					augmented.Set(k, j, kVal-factor*iVal)
				}
			}
		}
	}

	// Extract right side (inverse matrix)
	result = NewZeroMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			val, _ := augmented.Get(i, n+j)
			result.Set(i, j, val)
		}
	}

	return
}

// IsDiagonal checks if the matrix is a diagonal matrix (all non-diagonal elements are zero).
func (m Matrix) IsDiagonal() (isDiagonal bool) {
	if m.Rows != m.Cols {
		isDiagonal = false
		return
	} // Diagonal matrices must be square

	const epsilon = 1e-9
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if i != j {
				val, _ := m.Get(i, j)
				if math.Abs(val) > epsilon {
					isDiagonal = false
					return
				}
			}
		}
	}
	isDiagonal = true
	return
}

// IsSymmetric checks if the matrix is symmetric (A = A^T).
func (m Matrix) IsSymmetric() (isSymmetric bool) {
	if m.Rows != m.Cols {
		isSymmetric = false
		return
	}

	const epsilon = 1e-9
	for i := 0; i < m.Rows; i++ {
		for j := i + 1; j < m.Cols; j++ { // Check only upper part above diagonal
			val_ij, _ := m.Get(i, j)
			val_ji, _ := m.Get(j, i)
			if math.Abs(val_ij-val_ji) > epsilon {
				isSymmetric = false
				return
			}
		}
	}
	isSymmetric = true
	return
}

// IsUpperTriangular checks if the matrix is upper triangular (all elements below diagonal are zero).
func (m Matrix) IsUpperTriangular() (isUpperTriangular bool) {
	if m.Rows != m.Cols {
		isUpperTriangular = false
		return
	}

	const epsilon = 1e-9
	for i := 1; i < m.Rows; i++ { // From row 1
		for j := 0; j < i; j++ { // Check elements below diagonal
			val, _ := m.Get(i, j)
			if math.Abs(val) > epsilon {
				isUpperTriangular = false
				return
			}
		}
	}
	isUpperTriangular = true
	return
}

// IsLowerTriangular checks if the matrix is lower triangular (all elements above diagonal are zero).
func (m Matrix) IsLowerTriangular() (isLowerTriangular bool) {
	if m.Rows != m.Cols {
		isLowerTriangular = false
		return
	}

	const epsilon = 1e-9
	for i := 0; i < m.Rows; i++ {
		for j := i + 1; j < m.Cols; j++ { // Check elements above diagonal
			val, _ := m.Get(i, j)
			if math.Abs(val) > epsilon {
				isLowerTriangular = false
				return
			}
		}
	}
	isLowerTriangular = true
	return
}

/*
// --- Advanced linear algebra function skeletons (implementation omitted) ---
*/

// PseudoInverse computes the Moore-Penrose pseudoinverse (A+).
// This is generally calculated using SVD (Singular Value Decomposition).
/*
func (m Matrix) PseudoInverse() (result Matrix, err error) {
    // Actually requires SVD (Singular Value Decomposition) implementation
    err = fmt.Errorf("pseudoinverse calculation is not implemented (SVD required)")
    return
}
*/

// EigenSystem calculates eigenvalues and eigenvectors.
/*
func (m Matrix) EigenSystem() (eigenValues []complex128, eigenVectors []Vector, err error) {
    // Actually requires iterative algorithms like QR method or Jacobi method
    err = fmt.Errorf("eigenvalue and eigenvector calculation is not implemented")
    return
}
*/
