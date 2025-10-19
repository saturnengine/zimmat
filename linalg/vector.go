package linalg

import (
	"fmt"
)

// Vector represents an n-dimensional vector optimized for vector operations.
// It is implemented internally using a Tensor for consistency and performance.
//
// Vector provides a high-level interface for common vector operations such as
// addition, subtraction, scaling, dot product, normalization, and length calculation.
//
// Fields:
//   - tensor: Internal 1-dimensional tensor
//   - Data: Public field for compatibility (reference to tensor's Data)
//   - Dim: Number of dimensions
//
// Example:
//
//	v := linalg.NewVector(3.0, 4.0, 0.0)
//	length := v.Length() // 5.0
//	normalized, err := v.Normalize()
//	if err != nil {
//		panic(err)
//	}
type Vector struct {
	tensor *Tensor    // Internal tensor (1-dimensional)
	Data   []float64  // Public field for compatibility (reference to tensor's Data)
	Dim    int        // Number of dimensions
}

// NewVector creates a new Vector from the specified elements.
// If no elements are provided, returns an empty vector.
//
// Parameters:
//   - elements: Variable number of float64 values representing vector components
//
// Returns:
//   - Vector: A new vector containing the specified elements
//
// Example:
//
//	v1 := linalg.NewVector(1.0, 2.0, 3.0) // 3D vector
//	v2 := linalg.NewVector(0.0, 1.0)      // 2D vector
//	empty := linalg.NewVector()           // empty vector
func NewVector(elements ...float64) Vector {
	if len(elements) == 0 {
		return Vector{
			tensor: NewTensor(),
			Data:   []float64{},
			Dim:    0,
		}
	}
	
	tensor, err := NewTensorWithData(elements, len(elements))
	if err != nil {
		panic(fmt.Sprintf("ベクトル作成エラー: %v", err))
	}
	
	return Vector{
		tensor: tensor,
		Data:   tensor.Data, // テンソルのDataへの参照
		Dim:    len(elements),
	}
}

// NewVectorFromTensor creates a Vector from an existing Tensor.
// The tensor must be 1-dimensional (vector).
//
// Parameters:
//   - tensor: A 1-dimensional tensor to convert
//
// Returns:
//   - Vector: The new vector
//   - error: Non-nil if the tensor is not 1-dimensional
//
// Example:
//
//	t := linalg.NewTensor(3)
//	t.Set(1.0, 0)
//	t.Set(2.0, 1)
//	t.Set(3.0, 2)
//	v, err := linalg.NewVectorFromTensor(t)
//	if err != nil {
//		panic(err)
//	}
func NewVectorFromTensor(tensor *Tensor) (Vector, error) {
	if !tensor.IsVector() {
		return Vector{}, fmt.Errorf("テンソル（形状: %v）はベクトルではありません", tensor.Shape)
	}
	
	return Vector{
		tensor: tensor.Clone(),
		Data:   tensor.Data,
		Dim:    tensor.Shape[0],
	}, nil
}

// Add returns a new vector that is the element-wise sum of this vector
// and another vector. The vectors must have the same dimensions.
//
// Parameters:
//   - other: The vector to add to this vector
//
// Returns:
//   - Vector: A new vector containing the sum
//   - error: Non-nil if the vectors have different dimensions
//
// Example:
//
//	v1 := linalg.NewVector(1.0, 2.0, 3.0)
//	v2 := linalg.NewVector(4.0, 5.0, 6.0)
//	result, err := v1.Add(v2)
//	// result contains [5.0, 7.0, 9.0]
func (v Vector) Add(other Vector) (Vector, error) {
	result, err := v.tensor.Add(other.tensor)
	if err != nil {
		return Vector{}, fmt.Errorf("ベクトル加算エラー: %v", err)
	}
	
	vectorResult, err := NewVectorFromTensor(result)
	if err != nil {
		return Vector{}, fmt.Errorf("結果ベクトル作成エラー: %v", err)
	}
	
	return vectorResult, nil
}

// Subtract returns a new vector that is the element-wise difference of this vector
// and another vector. The vectors must have the same dimensions.
//
// Parameters:
//   - other: The vector to subtract from this vector
//
// Returns:
//   - Vector: A new vector containing the difference
//   - error: Non-nil if the vectors have different dimensions
//
// Example:
//
//	v1 := linalg.NewVector(5.0, 7.0, 9.0)
//	v2 := linalg.NewVector(1.0, 2.0, 3.0)
//	result, err := v1.Subtract(v2)
//	// result contains [4.0, 5.0, 6.0]
func (v Vector) Subtract(other Vector) (Vector, error) {
	result, err := v.tensor.Subtract(other.tensor)
	if err != nil {
		return Vector{}, fmt.Errorf("ベクトル減算エラー: %v", err)
	}
	
	vectorResult, err := NewVectorFromTensor(result)
	if err != nil {
		return Vector{}, fmt.Errorf("結果ベクトル作成エラー: %v", err)
	}
	
	return vectorResult, nil
}

// Scale returns a new vector that is this vector multiplied by a scalar value.
// Each component of the vector is multiplied by the scalar.
//
// Parameters:
//   - scalar: The scalar value to multiply by
//
// Returns:
//   - Vector: A new vector with scaled components
//
// Example:
//
//	v := linalg.NewVector(1.0, 2.0, 3.0)
//	scaled := v.Scale(2.5)
//	// scaled contains [2.5, 5.0, 7.5]
func (v Vector) Scale(scalar float64) Vector {
	result := v.tensor.Scale(scalar)
	
	vectorResult, err := NewVectorFromTensor(result)
	if err != nil {
		panic(fmt.Sprintf("スケール結果ベクトル作成エラー: %v", err))
	}
	
	return vectorResult
}

// Dot calculates the dot product (inner product) of two vectors.
// The vectors must have the same dimensions.
//
// Parameters:
//   - other: The vector to compute the dot product with
//
// Returns:
//   - float64: The dot product value
//   - error: Non-nil if the vectors have different dimensions
//
// The dot product is calculated as the sum of products of corresponding components.
//
// Example:
//
//	v1 := linalg.NewVector(1.0, 2.0, 3.0)
//	v2 := linalg.NewVector(4.0, 5.0, 6.0)
//	dot, err := v1.Dot(v2)
//	// dot = 1*4 + 2*5 + 3*6 = 32.0
func (v Vector) Dot(other Vector) (float64, error) {
	return v.tensor.VectorDot(other.tensor)
}

// LengthSq returns the squared length (squared norm) of the vector.
// This is more efficient than Length() when only comparing lengths
// or when the actual length is not needed.
//
// Returns:
//   - float64: The squared length of the vector
//
// The squared length is calculated as the dot product of the vector with itself.
//
// Example:
//
//	v := linalg.NewVector(3.0, 4.0)
//	lengthSq := v.LengthSq() // 25.0 (3² + 4²)
func (v Vector) LengthSq() float64 {
	dot, err := v.Dot(v)
	if err != nil {
		return 0 // エラーが発生した場合は0を返す
	}
	return dot
}

// Length returns the Euclidean length (L2 norm) of the vector.
//
// Returns:
//   - float64: The length of the vector (0 if error occurs)
//
// The length is calculated as the square root of the sum of squared components.
//
// Example:
//
//	v := linalg.NewVector(3.0, 4.0)
//	length := v.Length() // 5.0 (√(3² + 4²))
func (v Vector) Length() float64 {
	length, err := v.tensor.VectorLength()
	if err != nil {
		return 0 // エラーが発生した場合は0を返す
	}
	return length
}

// Normalize returns a new unit vector (length 1) in the same direction as this vector.
// This operation divides each component by the vector's length.
//
// Returns:
//   - Vector: A new normalized vector
//   - error: Non-nil if the vector is zero-length (cannot be normalized)
//
// A normalized vector maintains the same direction but has unit length.
//
// Example:
//
//	v := linalg.NewVector(3.0, 4.0)
//	unit, err := v.Normalize()
//	// unit contains [0.6, 0.8] with length 1.0
func (v Vector) Normalize() (Vector, error) {
	normalized, err := v.tensor.VectorNormalize()
	if err != nil {
		return Vector{}, fmt.Errorf("ベクトル正規化エラー: %v", err)
	}
	
	vectorResult, err := NewVectorFromTensor(normalized)
	if err != nil {
		return Vector{}, fmt.Errorf("正規化結果ベクトル作成エラー: %v", err)
	}
	
	return vectorResult, nil
}

// Get retrieves the element at the specified index.
//
// Parameters:
//   - index: The index of the element to retrieve (0-based)
//
// Returns:
//   - float64: The value at the specified index
//   - error: Non-nil if the index is out of bounds
//
// Example:
//
//	v := linalg.NewVector(1.0, 2.0, 3.0)
//	val, err := v.Get(1) // Returns 2.0
func (v Vector) Get(index int) (float64, error) {
	return v.tensor.Get(index)
}

// Set assigns a value to the element at the specified index.
//
// Parameters:
//   - index: The index of the element to set (0-based)
//   - value: The value to assign
//
// Returns:
//   - error: Non-nil if the index is out of bounds
//
// Example:
//
//	v := linalg.NewVector(1.0, 2.0, 3.0)
//	err := v.Set(1, 5.0) // Sets element at index 1 to 5.0
func (v Vector) Set(index int, value float64) error {
	err := v.tensor.Set(value, index)
	if err != nil {
		return err
	}
	// Dataフィールドも更新（参照なので自動的に更新されるが、明示的に保証）
	v.Data = v.tensor.Data
	return nil
}

// AsTensor returns a copy of this vector as a Tensor.
// The returned tensor is independent and can be modified without affecting the original vector.
//
// Returns:
//   - *Tensor: A 1-dimensional tensor containing the vector's data
//
// Example:
//
//	v := linalg.NewVector(1.0, 2.0, 3.0)
//	t := v.AsTensor()
//	// t is a 1D tensor with shape [3]
func (v Vector) AsTensor() *Tensor {
	return v.tensor.Clone()
}

// Clone creates a complete deep copy of the vector.
// The returned vector is independent and can be modified without affecting the original.
//
// Returns:
//   - Vector: A new vector with copied data
//
// Example:
//
//	original := linalg.NewVector(1.0, 2.0, 3.0)
//	copy := original.Clone()
//	copy.Set(0, 10.0) // Doesn't affect original
func (v Vector) Clone() Vector {
	clonedTensor := v.tensor.Clone()
	return Vector{
		tensor: clonedTensor,
		Data:   clonedTensor.Data,
		Dim:    v.Dim,
	}
}