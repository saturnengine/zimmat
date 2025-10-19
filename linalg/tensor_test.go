package linalg_test

import (
	"testing"

	"github.com/saturnengine/zimmat/linalg"
	"github.com/stretchr/testify/assert"
)

// TestNewTensor tests the NewTensor function with table-driven tests.
func TestNewTensor(t *testing.T) {
	tests := []struct {
		name          string
		dimensions    []int
		expectedRank  int
		expectedShape []int
		expectedSize  int
	}{
		{
			name:          "3-dimensional tensor (2x3x4)",
			dimensions:    []int{2, 3, 4},
			expectedRank:  3,
			expectedShape: []int{2, 3, 4},
			expectedSize:  24,
		},
		{
			name:          "2-dimensional tensor (3x3)",
			dimensions:    []int{3, 3},
			expectedRank:  2,
			expectedShape: []int{3, 3},
			expectedSize:  9,
		},
		{
			name:          "1-dimensional tensor (5)",
			dimensions:    []int{5},
			expectedRank:  1,
			expectedShape: []int{5},
			expectedSize:  5,
		},
		{
			name:          "empty tensor",
			dimensions:    []int{},
			expectedRank:  0,
			expectedShape: []int{},
			expectedSize:  0,
		},
		{
			name:          "zero dimension tensor",
			dimensions:    []int{2, 0, 3},
			expectedRank:  0,
			expectedShape: []int{},
			expectedSize:  0,
		},
		{
			name:          "negative dimension tensor",
			dimensions:    []int{2, -1, 3},
			expectedRank:  0,
			expectedShape: []int{},
			expectedSize:  0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := linalg.NewTensor(tt.dimensions...)
			assert.Equal(t, tt.expectedRank, tensor.Rank())
			assert.Equal(t, tt.expectedShape, tensor.Shape())
			assert.Equal(t, tt.expectedSize, tensor.Size())
		})
	}
}

// TestNewTensorWithData tests the NewTensorWithData function with table-driven tests.
func TestNewTensorWithData(t *testing.T) {
	tests := []struct {
		name       string
		data       []float64
		dimensions []int
		wantErr    bool
		checkValue bool
		getIndex   []int
		expected   float64
	}{
		{
			name:       "valid 2x3 tensor",
			data:       []float64{1, 2, 3, 4, 5, 6},
			dimensions: []int{2, 3},
			wantErr:    false,
			checkValue: true,
			getIndex:   []int{1, 2},
			expected:   6.0,
		},
		{
			name:       "empty shape with data",
			data:       []float64{1, 2, 3},
			dimensions: []int{},
			wantErr:    false,
			checkValue: false,
		},
		{
			name:       "negative dimension",
			data:       []float64{1, 2, 3},
			dimensions: []int{2, -1},
			wantErr:    true,
		},
		{
			name:       "zero dimension",
			data:       []float64{1, 2, 3},
			dimensions: []int{2, 0},
			wantErr:    true,
		},
		{
			name:       "mismatched data size",
			data:       []float64{1, 2, 3},
			dimensions: []int{2, 2},
			wantErr:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, err := linalg.NewTensorWithData(tt.data, tt.dimensions...)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			if tt.checkValue {
				val, err := tensor.Get(tt.getIndex...)
				assert.NoError(t, err)
				assert.True(t, almostEqual(val, tt.expected))
			}
		})
	}
}

// TestTensorGetSet tests the Get/Set methods with table-driven tests.
func TestTensorGetSet(t *testing.T) {
	tensor := linalg.NewTensor(3, 3)

	setTests := []struct {
		name    string
		value   float64
		indices []int
		wantErr bool
	}{
		{"valid set", 42.0, []int{1, 2}, false},
		{"wrong number of indices (too few)", 1.0, []int{1}, true},
		{"wrong number of indices (too many)", 1.0, []int{1, 2, 3}, true},
		{"negative index", 1.0, []int{-1, 0}, true},
		{"index too large (row)", 1.0, []int{3, 0}, true},
		{"index too large (col)", 1.0, []int{0, 3}, true},
	}

	for _, tt := range setTests {
		t.Run(tt.name, func(t *testing.T) {
			err := tensor.Set(tt.value, tt.indices...)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			// Verify the value was set
			val, _ := tensor.Get(tt.indices...)
			assert.True(t, almostEqual(val, tt.value))
		})
	}

	getTests := []struct {
		name    string
		indices []int
		wantErr bool
	}{
		{"valid get", []int{1, 2}, false},
		{"wrong number of indices (too few)", []int{1}, true},
		{"wrong number of indices (too many)", []int{1, 2, 3}, true},
		{"negative index", []int{-1, 0}, true},
		{"index too large (row)", []int{3, 0}, true},
		{"index too large (col)", []int{0, 3}, true},
	}

	for _, tt := range getTests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := tensor.Get(tt.indices...)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				assert.NoError(t, err)
			}
		})
	}
}

// TestTensorArithmeticOperations tests Add, Subtract, and Scale operations with table-driven tests.
func TestTensorArithmeticOperations(t *testing.T) {
	data1 := []float64{1, 2, 3, 4}
	data2 := []float64{5, 6, 7, 8}
	tensor1, _ := linalg.NewTensorWithData(data1, 2, 2)
	tensor2, _ := linalg.NewTensorWithData(data2, 2, 2)

	tests := []struct {
		name      string
		operation string
		t1        *linalg.Tensor
		t2        *linalg.Tensor
		scalar    float64
		expected  []float64
		wantErr   bool
	}{
		{
			name:      "valid addition",
			operation: "add",
			t1:        tensor1,
			t2:        tensor2,
			expected:  []float64{6, 8, 10, 12},
			wantErr:   false,
		},
		{
			name:      "valid subtraction",
			operation: "subtract",
			t1:        tensor2,
			t2:        tensor1,
			expected:  []float64{4, 4, 4, 4},
			wantErr:   false,
		},
		{
			name:      "scaling by 2.0",
			operation: "scale",
			t1:        tensor1,
			scalar:    2.0,
			expected:  []float64{2, 4, 6, 8},
			wantErr:   false,
		},
		{
			name:      "scaling by zero",
			operation: "scale",
			t1:        tensor1,
			scalar:    0.0,
			expected:  []float64{0, 0, 0, 0},
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var result *linalg.Tensor
			var err error

			switch tt.operation {
			case "add":
				result, err = tt.t1.Add(tt.t2)
			case "subtract":
				result, err = tt.t1.Subtract(tt.t2)
			case "scale":
				result = tt.t1.Scale(tt.scalar)
			}

			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			if tt.operation != "scale" {
				assert.NoError(t, err)
			}

			for i, expectedVal := range tt.expected {
				assert.True(t, almostEqual(result.Data()[i], expectedVal))
			}
		})
	}

	// Test shape mismatch errors
	tensor3x2 := linalg.NewTensor(3, 2)
	t.Run("add shape mismatch", func(t *testing.T) {
		_, err := tensor1.Add(tensor3x2)
		assert.Error(t, err)
	})

	t.Run("subtract shape mismatch", func(t *testing.T) {
		_, err := tensor1.Subtract(tensor3x2)
		assert.Error(t, err)
	})
}

// TestTensorReshape tests the Reshape operation with table-driven tests.
func TestTensorReshape(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, _ := linalg.NewTensorWithData(data, 2, 3)

	tests := []struct {
		name         string
		newShape     []int
		expectedRank int
		expectedSize int
		checkValue   bool
		getIndex     []int
		expectedVal  float64
		wantErr      bool
	}{
		{
			name:         "valid reshape 3x2",
			newShape:     []int{3, 2},
			expectedRank: 2,
			expectedSize: 6,
			checkValue:   true,
			getIndex:     []int{1, 1},
			expectedVal:  4.0,
			wantErr:      false,
		},
		{
			name:     "negative dimension",
			newShape: []int{2, -1},
			wantErr:  true,
		},
		{
			name:     "zero dimension",
			newShape: []int{2, 0},
			wantErr:  true,
		},
		{
			name:     "incompatible size",
			newShape: []int{2, 4},
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			reshaped, err := tensor.Reshape(tt.newShape...)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedRank, reshaped.Rank())
			assert.Equal(t, tt.expectedSize, reshaped.Size())
			if tt.checkValue {
				val, _ := reshaped.Get(tt.getIndex...)
				assert.True(t, almostEqual(val, tt.expectedVal))
			}
		})
	}
}

// TestTensorTranspose tests the transpose operation with table-driven tests.
func TestTensorTranspose(t *testing.T) {
	tests := []struct {
		name          string
		originalShape []int
		data          []float64
		expectedShape []int
		wantErr       bool
	}{
		{
			name:          "2x3 transpose",
			originalShape: []int{2, 3},
			data:          []float64{1, 2, 3, 4, 5, 6},
			expectedShape: []int{3, 2},
			wantErr:       false,
		},
		{
			name:          "square matrix transpose",
			originalShape: []int{2, 2},
			data:          []float64{1, 2, 3, 4},
			expectedShape: []int{2, 2},
			wantErr:       false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor, _ := linalg.NewTensorWithData(tt.data, tt.originalShape...)
			transposed, err := tensor.Transpose()
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.Equal(t, tt.expectedShape, transposed.Shape())

			// Check transpose correctness
			originalVal, _ := tensor.Get(0, 1)
			transposedVal, _ := transposed.Get(1, 0)
			assert.True(t, almostEqual(originalVal, transposedVal))
		})
	}

	// Test error cases
	errorTests := []struct {
		name  string
		shape []int
	}{
		{"1D tensor", []int{5}},
		{"3D tensor", []int{2, 3, 4}},
	}

	for _, tt := range errorTests {
		t.Run(tt.name+" error", func(t *testing.T) {
			tensor := linalg.NewTensor(tt.shape...)
			_, err := tensor.Transpose()
			assert.Error(t, err)
		})
	}
}

// TestTensorMatrixMultiply tests matrix multiplication with table-driven tests.
func TestTensorMatrixMultiply(t *testing.T) {
	tests := []struct {
		name     string
		dataA    []float64
		shapeA   []int
		dataB    []float64
		shapeB   []int
		expected []float64
		wantErr  bool
	}{
		{
			name:     "2x3 * 3x2 multiplication",
			dataA:    []float64{1, 2, 3, 4, 5, 6},
			shapeA:   []int{2, 3},
			dataB:    []float64{7, 8, 9, 10, 11, 12},
			shapeB:   []int{3, 2},
			expected: []float64{58, 64, 139, 154},
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensorA, _ := linalg.NewTensorWithData(tt.dataA, tt.shapeA...)
			tensorB, _ := linalg.NewTensorWithData(tt.dataB, tt.shapeB...)
			result, err := tensorA.MatrixMultiply(tensorB)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			for i, expectedVal := range tt.expected {
				assert.True(t, almostEqual(result.Data()[i], expectedVal))
			}
		})
	}

	// Test error cases
	tensor2d, _ := linalg.NewTensorWithData([]float64{1, 2, 3, 4}, 2, 2)
	tensor1d := linalg.NewTensor(3)
	tensor3d := linalg.NewTensor(2, 3, 4)

	errorTests := []struct {
		name string
		t1   *linalg.Tensor
		t2   *linalg.Tensor
	}{
		{"1D * 2D", tensor1d, tensor2d},
		{"2D * 1D", tensor2d, tensor1d},
		{"3D * 2D", tensor3d, tensor2d},
	}

	for _, tt := range errorTests {
		t.Run(tt.name+" error", func(t *testing.T) {
			_, err := tt.t1.MatrixMultiply(tt.t2)
			assert.Error(t, err)
		})
	}

	// Test incompatible dimensions
	t.Run("incompatible dimensions", func(t *testing.T) {
		tensor2x3, _ := linalg.NewTensorWithData([]float64{1, 2, 3, 4, 5, 6}, 2, 3)
		tensor4x2, _ := linalg.NewTensorWithData([]float64{1, 2, 3, 4, 5, 6, 7, 8}, 4, 2)
		_, err := tensor2x3.MatrixMultiply(tensor4x2)
		assert.Error(t, err)
	})
}

// TestTensorVectorOperations tests vector operations with table-driven tests.
func TestTensorVectorOperations(t *testing.T) {
	tests := []struct {
		name      string
		data1     []float64
		data2     []float64
		shape     []int
		operation string
		expected  float64
		wantErr   bool
	}{
		{
			name:      "dot product",
			data1:     []float64{1, 2, 3},
			data2:     []float64{4, 5, 6},
			shape:     []int{3},
			operation: "dot",
			expected:  32.0, // 1*4 + 2*5 + 3*6 = 32
			wantErr:   false,
		},
		{
			name:      "vector length",
			data1:     []float64{3, 4},
			shape:     []int{2},
			operation: "length",
			expected:  5.0,
			wantErr:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor1, _ := linalg.NewTensorWithData(tt.data1, tt.shape...)

			switch tt.operation {
			case "dot":
				tensor2, _ := linalg.NewTensorWithData(tt.data2, tt.shape...)
				result, err := tensor1.VectorDot(tensor2)
				if tt.wantErr {
					assert.Error(t, err)
					return
				}
				assert.NoError(t, err)
				assert.True(t, almostEqual(result, tt.expected))
			case "length":
				result, err := tensor1.VectorLength()
				if tt.wantErr {
					assert.Error(t, err)
					return
				}
				assert.NoError(t, err)
				assert.True(t, almostEqual(result, tt.expected))
			}
		})
	}

	// Test vector normalization
	t.Run("vector normalization", func(t *testing.T) {
		data := []float64{3, 4}
		tensor, _ := linalg.NewTensorWithData(data, 2)
		normalized, err := tensor.VectorNormalize()
		assert.NoError(t, err)

		// Check normalized length
		length, _ := normalized.VectorLength()
		assert.True(t, almostEqual(length, 1.0))

		// Check normalized values
		val0, _ := normalized.Get(0)
		val1, _ := normalized.Get(1)
		assert.True(t, almostEqual(val0, 0.6))
		assert.True(t, almostEqual(val1, 0.8))
	})

	// Test error cases for vector operations
	tensor2d := linalg.NewTensor(2, 3)
	tensor3d := linalg.NewTensor(2, 3, 4)

	vectorErrorTests := []struct {
		name      string
		tensor    *linalg.Tensor
		operation string
	}{
		{"2D tensor dot", tensor2d, "dot"},
		{"3D tensor dot", tensor3d, "dot"},
		{"2D tensor length", tensor2d, "length"},
		{"3D tensor length", tensor3d, "length"},
		{"2D tensor normalize", tensor2d, "normalize"},
		{"3D tensor normalize", tensor3d, "normalize"},
	}

	for _, tt := range vectorErrorTests {
		t.Run(tt.name+" error", func(t *testing.T) {
			switch tt.operation {
			case "dot":
				tensor1d := linalg.NewTensor(3)
				_, err := tt.tensor.VectorDot(tensor1d)
				assert.Error(t, err)
			case "length":
				_, err := tt.tensor.VectorLength()
				assert.Error(t, err)
			case "normalize":
				_, err := tt.tensor.VectorNormalize()
				assert.Error(t, err)
			}
		})
	}

	// Test zero vector normalization error
	t.Run("zero vector normalization error", func(t *testing.T) {
		zeroVector, _ := linalg.NewTensorWithData([]float64{0, 0, 0}, 3)
		_, err := zeroVector.VectorNormalize()
		assert.Error(t, err)
	})

	// Test dot product dimension mismatch
	t.Run("dot product dimension mismatch", func(t *testing.T) {
		tensor1d3 := linalg.NewTensor(3)
		tensor1d5 := linalg.NewTensor(5)
		_, err := tensor1d3.VectorDot(tensor1d5)
		assert.Error(t, err)
	})
}

// TestTensorTypeChecking tests IsVector and IsMatrix methods with table-driven tests.
func TestTensorTypeChecking(t *testing.T) {
	tests := []struct {
		name     string
		shape    []int
		isVector bool
		isMatrix bool
	}{
		{"1D tensor (vector)", []int{5}, true, false},
		{"2D tensor (matrix)", []int{3, 4}, false, true},
		{"3D tensor", []int{2, 3, 4}, false, false},
		{"0D tensor (scalar)", []int{}, false, false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := linalg.NewTensor(tt.shape...)
			assert.Equal(t, tt.isVector, tensor.IsVector())
			assert.Equal(t, tt.isMatrix, tensor.IsMatrix())
		})
	}
}

// TestTensorConversions tests AsVector and AsMatrix methods with table-driven tests.
func TestTensorConversions(t *testing.T) {
	tests := []struct {
		name         string
		data         []float64
		shape        []int
		toVector     bool
		toMatrix     bool
		expectedDim  int
		expectedRows int
		expectedCols int
		wantErr      bool
	}{
		{
			name:        "1D to vector",
			data:        []float64{1, 2, 3},
			shape:       []int{3},
			toVector:    true,
			expectedDim: 3,
			wantErr:     false,
		},
		{
			name:         "2D to matrix",
			data:         []float64{1, 2, 3, 4},
			shape:        []int{2, 2},
			toMatrix:     true,
			expectedRows: 2,
			expectedCols: 2,
			wantErr:      false,
		},
		{
			name:     "2D to vector (error)",
			shape:    []int{2, 3},
			toVector: true,
			wantErr:  true,
		},
		{
			name:     "1D to matrix (error)",
			shape:    []int{5},
			toMatrix: true,
			wantErr:  true,
		},
		{
			name:     "3D to vector (error)",
			shape:    []int{2, 3, 4},
			toVector: true,
			wantErr:  true,
		},
		{
			name:     "3D to matrix (error)",
			shape:    []int{2, 3, 4},
			toMatrix: true,
			wantErr:  true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var tensor *linalg.Tensor
			if tt.data != nil {
				tensor, _ = linalg.NewTensorWithData(tt.data, tt.shape...)
			} else {
				tensor = linalg.NewTensor(tt.shape...)
			}

			if tt.toVector {
				vector, err := tensor.AsVector()
				if tt.wantErr {
					assert.Error(t, err)
					return
				}
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedDim, vector.Dim())
			}

			if tt.toMatrix {
				matrix, err := tensor.AsMatrix()
				if tt.wantErr {
					assert.Error(t, err)
					return
				}
				assert.NoError(t, err)
				assert.Equal(t, tt.expectedRows, matrix.Rows())
				assert.Equal(t, tt.expectedCols, matrix.Cols())
			}
		})
	}
}

// TestTensorClone tests the Clone method with table-driven tests.
func TestTensorClone(t *testing.T) {
	tests := []struct {
		name  string
		data  []float64
		shape []int
	}{
		{
			name:  "2x2 tensor clone",
			data:  []float64{1, 2, 3, 4},
			shape: []int{2, 2},
		},
		{
			name:  "1D tensor clone",
			data:  []float64{1, 2, 3, 4, 5},
			shape: []int{5},
		},
		{
			name:  "empty tensor clone",
			data:  []float64{},
			shape: []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var original *linalg.Tensor
			if len(tt.data) > 0 {
				original, _ = linalg.NewTensorWithData(tt.data, tt.shape...)
			} else {
				original = linalg.NewTensor(tt.shape...)
			}

			cloned := original.Clone()

			// Check if clone has same data
			assert.Equal(t, original.Rank(), cloned.Rank())
			// For empty tensors, both shape slices might be nil, so check length instead
			if original.Rank() == 0 {
				assert.Equal(t, 0, len(cloned.Shape()))
			} else {
				assert.Equal(t, original.Shape(), cloned.Shape())
			}
			for i, val := range original.Data() {
				assert.True(t, almostEqual(val, cloned.Data()[i]))
			}

			// Verify independence (only test if tensor has elements)
			if original.Size() > 0 && original.Rank() > 0 {
				indices := make([]int, original.Rank())
				original.Set(99.0, indices...)
				clonedVal, _ := cloned.Get(indices...)
				assert.False(t, almostEqual(clonedVal, 99.0))
			}
		})
	}
}

// TestTensorStringAndSize tests String and Size methods with table-driven tests.
func TestTensorStringAndSize(t *testing.T) {
	tests := []struct {
		name         string
		shape        []int
		expectedSize int
	}{
		{"3D tensor", []int{2, 3, 4}, 24},
		{"2D tensor", []int{3, 3}, 9},
		{"1D tensor", []int{5}, 5},
		{"empty tensor", []int{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tensor := linalg.NewTensor(tt.shape...)

			// Test Size
			assert.Equal(t, tt.expectedSize, tensor.Size())

			// Test String (should not be empty and should contain "Tensor")
			str := tensor.String()
			assert.NotEmpty(t, str)
			assert.True(t, len(str) >= 6) // Should contain "Tensor" and other info
		})
	}
}
