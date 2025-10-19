// Package linalg provides comprehensive linear algebra operations including
// vectors, matrices, and tensors for high-performance mathematical computations.
//
// The package is built around three main types:
//   - Tensor: A generic N-dimensional array that serves as the foundation
//   - Vector: A 1-dimensional tensor optimized for vector operations
//   - Matrix: A 2-dimensional tensor optimized for matrix operations
//
// All operations are designed to be memory-efficient and provide robust
// error handling for dimension mismatches and invalid operations.
//
// Example usage:
//
//	// Create a 3D vector
//	v := linalg.NewVector(1.0, 2.0, 3.0)
//	length := v.Length()
//
//	// Create a 2x2 matrix
//	data := [][]float64{{1, 2}, {3, 4}}
//	m, err := linalg.NewMatrix(data)
//	if err != nil {
//		panic(err)
//	}
//
//	// Create a custom tensor
//	t := linalg.NewTensor(2, 3, 4) // 2x3x4 tensor
package linalg

import (
	"fmt"
	"math"
)

// Tensor represents a generic N-dimensional array that can handle arbitrary
// dimensions. It serves as the foundation for Vector (1D) and Matrix (2D)
// types, providing a unified interface for tensor operations.
//
// The tensor stores data in a flattened array using row-major (C-style) ordering
// for efficient memory access and cache performance.
//
// Fields:
//   - Data: Flattened data stored in row-major order
//   - Shape: Size of each dimension (e.g., [2, 3] = 2x3 matrix, [5] = 5D vector)
//   - Rank: Number of dimensions (tensor order)
//
// Example:
//
//	// Create a 2x3 tensor (matrix)
//	t := linalg.NewTensor(2, 3)
//	t.Set(1.5, 0, 1) // Set element at row 0, column 1
//	val, _ := t.Get(0, 1) // Get element at row 0, column 1
type Tensor struct {
	data  []float64 // Flattened data stored in row-major order
	shape []int     // Size of each dimension (e.g., [2, 3] = 2x3 matrix, [5] = 5D vector)
	rank  int       // Number of dimensions (tensor order)
}

// Data returns the flattened data stored in row-major order
func (t *Tensor) Data() []float64 { return t.data }

// Shape returns the size of each dimension
func (t *Tensor) Shape() []int { return t.shape }

// Rank returns the number of dimensions (tensor order)
func (t *Tensor) Rank() int { return t.rank }

// NewTensor creates a new tensor with the specified shape.
// All elements are initialized to zero.
//
// Parameters:
//   - shape: Variable number of integers specifying the size of each dimension
//
// Returns:
//   - *Tensor: A pointer to the newly created tensor
//
// Panics if any dimension is less than or equal to zero.
//
// Example:
//
//	t := linalg.NewTensor(2, 3)    // 2x3 matrix (6 elements)
//	v := linalg.NewTensor(5)       // 5D vector (5 elements)
//	cube := linalg.NewTensor(2, 2, 2) // 2x2x2 tensor (8 elements)
func NewTensor(shape ...int) (result *Tensor) {
	if len(shape) == 0 {
		result = &Tensor{data: []float64{}, shape: []int{}, rank: 0}
		return
	}

	// Calculate total number of elements
	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			// Return empty tensor instead of panicking
			result = &Tensor{data: []float64{}, shape: []int{}, rank: 0}
			return
		}
		size *= dim
	}

	result = &Tensor{
		data:  make([]float64, size),
		shape: append([]int(nil), shape...), // Copy slice
		rank:  len(shape),
	}
	return
}

// NewTensorWithData creates a new tensor with the specified data and shape.
// The provided data slice is copied to ensure tensor independence.
//
// Parameters:
//   - data: Flattened data array in row-major order
//   - shape: Variable number of integers specifying the size of each dimension
//
// Returns:
//   - *Tensor: A pointer to the newly created tensor
//   - error: Non-nil if data size doesn't match shape or if shape is invalid
//
// The total number of elements in data must exactly match the product
// of all dimensions in shape.
//
// Example:
//
//	data := []float64{1, 2, 3, 4, 5, 6}
//	t, err := linalg.NewTensorWithData(data, 2, 3) // 2x3 matrix
//	if err != nil {
//		panic(err)
//	}
func NewTensorWithData(data []float64, shape ...int) (result *Tensor, err error) {
	if len(shape) == 0 {
		result = &Tensor{data: []float64{}, shape: []int{}, rank: 0}
		return
	}

	// Calculate total number of elements
	expectedSize := 1
	for _, dim := range shape {
		if dim <= 0 {
			err = fmt.Errorf("tensor dimension must be positive: %d", dim)
			return
		}
		expectedSize *= dim
	}

	if len(data) != expectedSize {
		err = fmt.Errorf("data size (%d) does not match shape (%v). expected: %d",
			len(data), shape, expectedSize)
		return
	}

	result = &Tensor{
		data:  append([]float64(nil), data...), // Copy data
		shape: append([]int(nil), shape...),    // Copy shape
		rank:  len(shape),
	}
	return
}

// flatIndex converts multi-dimensional indices to a one-dimensional index.
func (t *Tensor) flatIndex(indices []int) (flatIdx int, err error) {
	if len(indices) != t.rank {
		err = fmt.Errorf("number of indices (%d) does not match tensor rank (%d)",
			len(indices), t.rank)
		return
	}

	multiplier := 1

	// Calculate index in row-major (C-style) order
	for i := t.rank - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.shape[i] {
			err = fmt.Errorf("index[%d]=%d is out of bounds (0-%d)",
				i, indices[i], t.shape[i]-1)
			return
		}
		flatIdx += indices[i] * multiplier
		multiplier *= t.shape[i]
	}

	return
}

// Get retrieves an element at the specified multi-dimensional indices.
//
// Parameters:
//   - indices: Variable number of integers specifying the position in each dimension
//
// Returns:
//   - float64: The value at the specified position
//   - error: Non-nil if indices are invalid or out of bounds
//
// The number of indices must match the tensor's rank, and each index
// must be within the valid range [0, dimension_size).
//
// Example:
//
//	t := linalg.NewTensor(2, 3) // 2x3 tensor
//	t.Set(42.0, 1, 2)          // Set element at row 1, column 2
//	val, err := t.Get(1, 2)    // Get element at row 1, column 2
//	// val == 42.0
func (t *Tensor) Get(indices ...int) (value float64, err error) {
	flatIdx, err := t.flatIndex(indices)
	if err != nil {
		return
	}
	value = t.data[flatIdx]
	return
}

// Set assigns a value to the element at the specified multi-dimensional indices.
//
// Parameters:
//   - value: The value to assign
//   - indices: Variable number of integers specifying the position in each dimension
//
// Returns:
//   - error: Non-nil if indices are invalid or out of bounds
//
// The number of indices must match the tensor's rank, and each index
// must be within the valid range [0, dimension_size).
//
// Example:
//
//	t := linalg.NewTensor(2, 3) // 2x3 tensor
//	err := t.Set(42.0, 1, 2)   // Set element at row 1, column 2
//	if err != nil {
//		panic(err)
//	}
func (t *Tensor) Set(value float64, indices ...int) (err error) {
	flatIdx, err := t.flatIndex(indices)
	if err != nil {
		return
	}
	t.data[flatIdx] = value
	return
}

// Size returns the total number of elements in the tensor.
// This is equivalent to the product of all dimensions in the shape.
//
// Returns:
//   - int: The total number of elements
//
// Example:
//
//	t := linalg.NewTensor(2, 3, 4) // 2x3x4 tensor
//	size := t.Size() // Returns 24 (2 * 3 * 4)
func (t *Tensor) Size() (size int) {
	size = len(t.data)
	return
}

// Clone creates a complete deep copy of the tensor.
// The returned tensor is independent of the original and can be
// modified without affecting the source tensor.
//
// Returns:
//   - *Tensor: A pointer to the new tensor with copied data and shape
//
// Example:
//
//	original := linalg.NewTensor(2, 3)
//	copy := original.Clone()
//	copy.Set(42.0, 0, 0) // Doesn't affect original
func (t *Tensor) Clone() (result *Tensor) {
	result = &Tensor{
		data:  append([]float64(nil), t.data...),
		shape: append([]int(nil), t.shape...),
		rank:  t.rank,
	}
	return
}

// Reshape changes the shape of the tensor. The total number of elements must be preserved.
func (t *Tensor) Reshape(newShape ...int) (result *Tensor, err error) {
	// Calculate total number of elements in new shape
	newSize := 1
	for _, dim := range newShape {
		if dim <= 0 {
			err = fmt.Errorf("new shape dimension must be positive: %d", dim)
			return
		}
		newSize *= dim
	}

	if newSize != len(t.data) {
		err = fmt.Errorf("new shape (%v) total elements (%d) does not match current data size (%d)",
			newShape, newSize, len(t.data))
		return
	}

	result = &Tensor{
		data:  append([]float64(nil), t.data...), // Copy data
		shape: append([]int(nil), newShape...),   // Copy new shape
		rank:  len(newShape),
	}
	return
}

// Add performs element-wise addition between tensors.
func (t *Tensor) Add(other *Tensor) (result *Tensor, err error) {
	if !t.sameShape(other) {
		err = fmt.Errorf("tensor shapes differ: %v and %v", t.shape, other.shape)
		return
	}

	result = t.Clone()
	for i := range result.data {
		result.data[i] += other.data[i]
	}

	return
}

// Subtract performs element-wise subtraction between tensors.
func (t *Tensor) Subtract(other *Tensor) (result *Tensor, err error) {
	if !t.sameShape(other) {
		err = fmt.Errorf("tensor shapes differ: %v and %v", t.shape, other.shape)
		return
	}

	result = t.Clone()
	for i := range result.data {
		result.data[i] -= other.data[i]
	}

	return
}

// Scale multiplies all tensor elements by a scalar value.
func (t *Tensor) Scale(scalar float64) (result *Tensor) {
	result = t.Clone()
	for i := range result.data {
		result.data[i] *= scalar
	}
	return
}

// sameShape checks if two tensors have the same shape.
func (t *Tensor) sameShape(other *Tensor) (isSame bool) {
	if t.rank != other.rank {
		isSame = false
		return
	}
	for i := range t.shape {
		if t.shape[i] != other.shape[i] {
			isSame = false
			return
		}
	}
	isSame = true
	return
}

// IsVector checks if the tensor is 1-dimensional (vector).
func (t *Tensor) IsVector() (isVector bool) {
	isVector = t.rank == 1
	return
}

// IsMatrix checks if the tensor is 2-dimensional (matrix).
func (t *Tensor) IsMatrix() (isMatrix bool) {
	isMatrix = t.rank == 2
	return
}

// AsVector treats the tensor as a Vector (1-dimensional only).
func (t *Tensor) AsVector() (result Vector, err error) {
	if !t.IsVector() {
		err = fmt.Errorf("tensor with shape %v cannot be converted to vector", t.shape)
		return
	}

	result = Vector{
		data: append([]float64(nil), t.data...), // Copy data
		dim:  t.shape[0],
	}
	return
}

// AsMatrix treats the tensor as a Matrix (2-dimensional only).
func (t *Tensor) AsMatrix() (result Matrix, err error) {
	if !t.IsMatrix() {
		err = fmt.Errorf("tensor with shape %v cannot be converted to matrix", t.shape)
		return
	}

	result = Matrix{
		data: append([]float64(nil), t.data...), // Copy data
		rows: t.shape[0],
		cols: t.shape[1],
	}
	return
}

// Transpose performs transposition of a 2-dimensional tensor (matrix).
func (t *Tensor) Transpose() (result *Tensor, err error) {
	if !t.IsMatrix() {
		err = fmt.Errorf("transpose can only be performed on 2-dimensional tensors. current shape: %v", t.shape)
		return
	}

	rows, cols := t.shape[0], t.shape[1]
	result = NewTensor(cols, rows) // Swap rows and columns

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val, _ := t.Get(i, j)
			result.Set(val, j, i) // Swap indices
		}
	}

	return
}

// MatrixMultiply performs matrix multiplication of two 2-dimensional tensors (matrices).
func (t *Tensor) MatrixMultiply(other *Tensor) (result *Tensor, err error) {
	if !t.IsMatrix() || !other.IsMatrix() {
		err = fmt.Errorf("matrix multiplication can only be performed between 2-dimensional tensors")
		return
	}

	if t.shape[1] != other.shape[0] {
		err = fmt.Errorf("matrix multiplication rule violation: left columns (%d) != right rows (%d)",
			t.shape[1], other.shape[0])
		return
	}

	resultRows, resultCols := t.shape[0], other.shape[1]
	result = NewTensor(resultRows, resultCols)

	for i := 0; i < resultRows; i++ {
		for j := 0; j < resultCols; j++ {
			var sum float64
			for k := 0; k < t.shape[1]; k++ {
				val1, _ := t.Get(i, k)
				val2, _ := other.Get(k, j)
				sum += val1 * val2
			}
			result.Set(sum, i, j)
		}
	}

	return
}

// VectorDot calculates the dot product of two 1-dimensional tensors (vectors).
func (t *Tensor) VectorDot(other *Tensor) (dotProduct float64, err error) {
	if !t.IsVector() || !other.IsVector() {
		err = fmt.Errorf("dot product can only be calculated between 1-dimensional tensors (vectors)")
		return
	}

	if t.shape[0] != other.shape[0] {
		err = fmt.Errorf("vector dimensions differ: %d and %d", t.shape[0], other.shape[0])
		return
	}

	for i := 0; i < t.shape[0]; i++ {
		dotProduct += t.data[i] * other.data[i]
	}

	return
}

// VectorLength calculates the length (norm) of a 1-dimensional tensor (vector).
func (t *Tensor) VectorLength() (length float64, err error) {
	if !t.IsVector() {
		err = fmt.Errorf("length calculation can only be performed on 1-dimensional tensors (vectors)")
		return
	}

	var sum float64
	for _, val := range t.data {
		sum += val * val
	}

	length = math.Sqrt(sum)
	return
}

// VectorNormalize normalizes a 1-dimensional tensor (vector).
func (t *Tensor) VectorNormalize() (result *Tensor, err error) {
	if !t.IsVector() {
		err = fmt.Errorf("normalization can only be performed on 1-dimensional tensors (vectors)")
		return
	}

	length, err := t.VectorLength()
	if err != nil {
		return
	}

	if length == 0 {
		err = fmt.Errorf("zero vector cannot be normalized")
		return
	}

	result = t.Scale(1.0 / length)
	return
}

// String returns the string representation of the tensor (for debugging).
func (t *Tensor) String() (str string) {
	str = fmt.Sprintf("Tensor{Shape: %v, Rank: %d, Size: %d}", t.shape, t.rank, len(t.data))
	return
}
