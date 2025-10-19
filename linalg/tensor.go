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
	Data  []float64 // Flattened data stored in row-major order
	Shape []int     // Size of each dimension (e.g., [2, 3] = 2x3 matrix, [5] = 5D vector)
	Rank  int       // Number of dimensions (tensor order)
}

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
func NewTensor(shape ...int) *Tensor {
	if len(shape) == 0 {
		return &Tensor{Data: []float64{}, Shape: []int{}, Rank: 0}
	}

	// 総要素数を計算
	size := 1
	for _, dim := range shape {
		if dim <= 0 {
			panic(fmt.Sprintf("テンソルの次元は正の値である必要があります: %d", dim))
		}
		size *= dim
	}

	return &Tensor{
		Data:  make([]float64, size),
		Shape: append([]int(nil), shape...), // スライスをコピー
		Rank:  len(shape),
	}
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
func NewTensorWithData(data []float64, shape ...int) (*Tensor, error) {
	if len(shape) == 0 {
		return &Tensor{Data: []float64{}, Shape: []int{}, Rank: 0}, nil
	}

	// 総要素数を計算
	expectedSize := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("テンソルの次元は正の値である必要があります: %d", dim)
		}
		expectedSize *= dim
	}

	if len(data) != expectedSize {
		return nil, fmt.Errorf("データサイズ(%d)が形状(%v)と一致しません。期待値: %d", 
			len(data), shape, expectedSize)
	}

	return &Tensor{
		Data:  append([]float64(nil), data...), // データをコピー
		Shape: append([]int(nil), shape...),    // 形状をコピー
		Rank:  len(shape),
	}, nil
}

// flatIndex は多次元インデックスを1次元インデックスに変換します。
func (t *Tensor) flatIndex(indices []int) (int, error) {
	if len(indices) != t.Rank {
		return 0, fmt.Errorf("インデックス数(%d)がテンソルの次元数(%d)と一致しません", 
			len(indices), t.Rank)
	}

	flatIdx := 0
	multiplier := 1
	
	// 行優先（C-style）でインデックスを計算
	for i := t.Rank - 1; i >= 0; i-- {
		if indices[i] < 0 || indices[i] >= t.Shape[i] {
			return 0, fmt.Errorf("インデックス[%d]=%d が範囲外です（0-%d）", 
				i, indices[i], t.Shape[i]-1)
		}
		flatIdx += indices[i] * multiplier
		multiplier *= t.Shape[i]
	}
	
	return flatIdx, nil
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
func (t *Tensor) Get(indices ...int) (float64, error) {
	flatIdx, err := t.flatIndex(indices)
	if err != nil {
		return 0, err
	}
	return t.Data[flatIdx], nil
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
func (t *Tensor) Set(value float64, indices ...int) error {
	flatIdx, err := t.flatIndex(indices)
	if err != nil {
		return err
	}
	t.Data[flatIdx] = value
	return nil
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
func (t *Tensor) Size() int {
	return len(t.Data)
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
func (t *Tensor) Clone() *Tensor {
	return &Tensor{
		Data:  append([]float64(nil), t.Data...),
		Shape: append([]int(nil), t.Shape...),
		Rank:  t.Rank,
	}
}

// Reshape はテンソルの形状を変更します。総要素数は保持される必要があります。
func (t *Tensor) Reshape(newShape ...int) (*Tensor, error) {
	// 新しい形状の総要素数を計算
	newSize := 1
	for _, dim := range newShape {
		if dim <= 0 {
			return nil, fmt.Errorf("新しい形状の次元は正の値である必要があります: %d", dim)
		}
		newSize *= dim
	}

	if newSize != len(t.Data) {
		return nil, fmt.Errorf("新しい形状(%v)の総要素数(%d)が現在のデータサイズ(%d)と一致しません",
			newShape, newSize, len(t.Data))
	}

	return &Tensor{
		Data:  append([]float64(nil), t.Data...), // データをコピー
		Shape: append([]int(nil), newShape...),   // 新しい形状をコピー
		Rank:  len(newShape),
	}, nil
}

// Add はテンソル同士の要素ごとの加算を行います。
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	if !t.sameShape(other) {
		return nil, fmt.Errorf("テンソルの形状が異なります: %v と %v", t.Shape, other.Shape)
	}

	result := t.Clone()
	for i := range result.Data {
		result.Data[i] += other.Data[i]
	}

	return result, nil
}

// Subtract はテンソル同士の要素ごとの減算を行います。
func (t *Tensor) Subtract(other *Tensor) (*Tensor, error) {
	if !t.sameShape(other) {
		return nil, fmt.Errorf("テンソルの形状が異なります: %v と %v", t.Shape, other.Shape)
	}

	result := t.Clone()
	for i := range result.Data {
		result.Data[i] -= other.Data[i]
	}

	return result, nil
}

// Scale はテンソルの全要素をスカラー値で乗算します。
func (t *Tensor) Scale(scalar float64) *Tensor {
	result := t.Clone()
	for i := range result.Data {
		result.Data[i] *= scalar
	}
	return result
}

// sameShape は2つのテンソルが同じ形状かどうかをチェックします。
func (t *Tensor) sameShape(other *Tensor) bool {
	if t.Rank != other.Rank {
		return false
	}
	for i := range t.Shape {
		if t.Shape[i] != other.Shape[i] {
			return false
		}
	}
	return true
}

// IsVector はテンソルが1次元（ベクトル）かどうかをチェックします。
func (t *Tensor) IsVector() bool {
	return t.Rank == 1
}

// IsMatrix はテンソルが2次元（行列）かどうかをチェックします。
func (t *Tensor) IsMatrix() bool {
	return t.Rank == 2
}

// AsVector はテンソルをVectorとして扱います（1次元の場合のみ）。
func (t *Tensor) AsVector() (Vector, error) {
	if !t.IsVector() {
		return Vector{}, fmt.Errorf("テンソル（形状: %v）をベクトルに変換できません", t.Shape)
	}
	
	return Vector{
		Data: append([]float64(nil), t.Data...), // データをコピー
		Dim:  t.Shape[0],
	}, nil
}

// AsMatrix はテンソルをMatrixとして扱います（2次元の場合のみ）。
func (t *Tensor) AsMatrix() (Matrix, error) {
	if !t.IsMatrix() {
		return Matrix{}, fmt.Errorf("テンソル（形状: %v）を行列に変換できません", t.Shape)
	}
	
	return Matrix{
		Data: append([]float64(nil), t.Data...), // データをコピー
		Rows: t.Shape[0],
		Cols: t.Shape[1],
	}, nil
}

// Transpose は2次元テンソル（行列）の転置を行います。
func (t *Tensor) Transpose() (*Tensor, error) {
	if !t.IsMatrix() {
		return nil, fmt.Errorf("転置は2次元テンソルに対してのみ実行できます。現在の形状: %v", t.Shape)
	}

	rows, cols := t.Shape[0], t.Shape[1]
	result := NewTensor(cols, rows) // 行と列を入れ替え

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			val, _ := t.Get(i, j)
			result.Set(val, j, i) // インデックスを入れ替え
		}
	}

	return result, nil
}

// MatrixMultiply は2つの2次元テンソル（行列）の行列乗算を行います。
func (t *Tensor) MatrixMultiply(other *Tensor) (*Tensor, error) {
	if !t.IsMatrix() || !other.IsMatrix() {
		return nil, fmt.Errorf("行列乗算は2次元テンソル同士でのみ実行できます")
	}

	if t.Shape[1] != other.Shape[0] {
		return nil, fmt.Errorf("行列乗算のルールに違反します: 左の列数(%d) != 右の行数(%d)",
			t.Shape[1], other.Shape[0])
	}

	resultRows, resultCols := t.Shape[0], other.Shape[1]
	result := NewTensor(resultRows, resultCols)

	for i := 0; i < resultRows; i++ {
		for j := 0; j < resultCols; j++ {
			var sum float64
			for k := 0; k < t.Shape[1]; k++ {
				val1, _ := t.Get(i, k)
				val2, _ := other.Get(k, j)
				sum += val1 * val2
			}
			result.Set(sum, i, j)
		}
	}

	return result, nil
}

// VectorDot は1次元テンソル（ベクトル）同士の内積を計算します。
func (t *Tensor) VectorDot(other *Tensor) (float64, error) {
	if !t.IsVector() || !other.IsVector() {
		return 0, fmt.Errorf("内積は1次元テンソル（ベクトル）同士でのみ計算できます")
	}

	if t.Shape[0] != other.Shape[0] {
		return 0, fmt.Errorf("ベクトルの次元数が異なります: %d と %d", t.Shape[0], other.Shape[0])
	}

	var sum float64
	for i := 0; i < t.Shape[0]; i++ {
		sum += t.Data[i] * other.Data[i]
	}

	return sum, nil
}

// VectorLength は1次元テンソル（ベクトル）の長さ（ノルム）を計算します。
func (t *Tensor) VectorLength() (float64, error) {
	if !t.IsVector() {
		return 0, fmt.Errorf("長さの計算は1次元テンソル（ベクトル）に対してのみ実行できます")
	}

	var sum float64
	for _, val := range t.Data {
		sum += val * val
	}

	return math.Sqrt(sum), nil
}

// VectorNormalize は1次元テンソル（ベクトル）を正規化します。
func (t *Tensor) VectorNormalize() (*Tensor, error) {
	if !t.IsVector() {
		return nil, fmt.Errorf("正規化は1次元テンソル（ベクトル）に対してのみ実行できます")
	}

	length, err := t.VectorLength()
	if err != nil {
		return nil, err
	}

	if length == 0 {
		return nil, fmt.Errorf("ゼロベクトルは正規化できません")
	}

	return t.Scale(1.0 / length), nil
}

// String はテンソルの文字列表現を返します（デバッグ用）。
func (t *Tensor) String() string {
	return fmt.Sprintf("Tensor{Shape: %v, Rank: %d, Size: %d}", t.Shape, t.Rank, len(t.Data))
}