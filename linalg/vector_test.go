package linalg_test

import (
	"testing"

	"github.com/saturnengine/zimmat/linalg"
	"github.com/stretchr/testify/assert"
)

// testVectorsEqual checks if two vectors are equal (elements and dimensions match within tolerance).
func testVectorsEqual(v1, v2 linalg.Vector) bool {
	if v1.Dim() != v2.Dim() {
		return false
	}
	for i := 0; i < v1.Dim(); i++ {
		if !almostEqual(v1.Data()[i], v2.Data()[i]) {
			return false
		}
	}
	return true
}

// TestNewVector tests the NewVector function with table-driven tests.
func TestNewVector(t *testing.T) {
	tests := []struct {
		name     string
		elements []float64
		expected []float64
		dim      int
	}{
		{
			name:     "3-dimensional vector",
			elements: []float64{1.0, 2.0, 3.0},
			expected: []float64{1.0, 2.0, 3.0},
			dim:      3,
		},
		{
			name:     "empty vector",
			elements: []float64{},
			expected: []float64{},
			dim:      0,
		},
		{
			name:     "single element",
			elements: []float64{42.0},
			expected: []float64{42.0},
			dim:      1,
		},
		{
			name:     "2-dimensional vector",
			elements: []float64{-1.5, 3.7},
			expected: []float64{-1.5, 3.7},
			dim:      2,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := linalg.NewVector(tt.elements...)
			assert.Equal(t, tt.dim, v.Dim())
			for i, expected := range tt.expected {
				assert.True(t, almostEqual(v.Data()[i], expected))
			}
		})
	}
}

// TestVectorAdd tests the vector addition method with table-driven tests.
func TestVectorAdd(t *testing.T) {
	tests := []struct {
		name     string
		v1       func() linalg.Vector
		v2       func() linalg.Vector
		expected func() linalg.Vector
		wantErr  bool
	}{
		{
			name:     "valid 3D addition",
			v1:       func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
			v2:       func() linalg.Vector { return linalg.NewVector(4, 5, 6) },
			expected: func() linalg.Vector { return linalg.NewVector(5, 7, 9) },
			wantErr:  false,
		},
		{
			name:    "dimension mismatch",
			v1:      func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
			v2:      func() linalg.Vector { return linalg.NewVector(1, 2) },
			wantErr: true,
		},
		{
			name:     "2D vectors",
			v1:       func() linalg.Vector { return linalg.NewVector(1.5, -2.3) },
			v2:       func() linalg.Vector { return linalg.NewVector(-0.5, 3.3) },
			expected: func() linalg.Vector { return linalg.NewVector(1.0, 1.0) },
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v1 := tt.v1()
			v2 := tt.v2()
			result, err := v1.Add(v2)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testVectorsEqual(result, tt.expected()))
		})
	}
}

// TestVectorSubtract tests the vector subtraction method with table-driven tests.
func TestVectorSubtract(t *testing.T) {
	tests := []struct {
		name     string
		v1       func() linalg.Vector
		v2       func() linalg.Vector
		expected func() linalg.Vector
		wantErr  bool
	}{
		{
			name:     "valid 3D subtraction",
			v1:       func() linalg.Vector { return linalg.NewVector(5, 7, 9) },
			v2:       func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
			expected: func() linalg.Vector { return linalg.NewVector(4, 5, 6) },
			wantErr:  false,
		},
		{
			name:    "dimension mismatch",
			v1:      func() linalg.Vector { return linalg.NewVector(5, 7, 9) },
			v2:      func() linalg.Vector { return linalg.NewVector(1, 2) },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v1 := tt.v1()
			v2 := tt.v2()
			result, err := v1.Subtract(v2)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testVectorsEqual(result, tt.expected()))
		})
	}
}

// TestVectorScale tests the vector scaling method with table-driven tests.
func TestVectorScale(t *testing.T) {
	tests := []struct {
		name     string
		vector   func() linalg.Vector
		scalar   float64
		expected func() linalg.Vector
	}{
		{
			name:     "scale by 2.5",
			vector:   func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
			scalar:   2.5,
			expected: func() linalg.Vector { return linalg.NewVector(2.5, 5.0, 7.5) },
		},
		{
			name:     "scale by zero",
			vector:   func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
			scalar:   0,
			expected: func() linalg.Vector { return linalg.NewVector(0, 0, 0) },
		},
		{
			name:     "scale by negative",
			vector:   func() linalg.Vector { return linalg.NewVector(1, -2) },
			scalar:   -3,
			expected: func() linalg.Vector { return linalg.NewVector(-3, 6) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := tt.vector()
			result := v.Scale(tt.scalar)
			assert.True(t, testVectorsEqual(result, tt.expected()))
		})
	}
}

// TestVectorDot tests the dot product method with table-driven tests.
func TestVectorDot(t *testing.T) {
	tests := []struct {
		name     string
		v1       func() linalg.Vector
		v2       func() linalg.Vector
		expected float64
		wantErr  bool
	}{
		{
			name:     "perpendicular vectors",
			v1:       func() linalg.Vector { return linalg.NewVector(1, 0, 0) },
			v2:       func() linalg.Vector { return linalg.NewVector(0, 1, 0) },
			expected: 0.0,
			wantErr:  false,
		},
		{
			name:     "self dot product",
			v1:       func() linalg.Vector { return linalg.NewVector(2, 3, 4) },
			v2:       func() linalg.Vector { return linalg.NewVector(2, 3, 4) },
			expected: 29.0, // 2*2 + 3*3 + 4*4 = 4 + 9 + 16 = 29
			wantErr:  false,
		},
		{
			name:    "dimension mismatch",
			v1:      func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
			v2:      func() linalg.Vector { return linalg.NewVector(1, 2) },
			wantErr: true,
		},
		{
			name:     "2D vectors dot product",
			v1:       func() linalg.Vector { return linalg.NewVector(1, 2) },
			v2:       func() linalg.Vector { return linalg.NewVector(3, 4) },
			expected: 11.0, // 1*3 + 2*4 = 3 + 8 = 11
			wantErr:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v1 := tt.v1()
			v2 := tt.v2()
			dot, err := v1.Dot(v2)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, almostEqual(dot, tt.expected))
		})
	}
}

// TestVectorLength tests the length calculation methods with table-driven tests.
func TestVectorLength(t *testing.T) {
	tests := []struct {
		name           string
		vector         func() linalg.Vector
		expectedLength float64
		expectedLenSq  float64
	}{
		{
			name:           "3-4 vector (length 5)",
			vector:         func() linalg.Vector { return linalg.NewVector(3, 4) },
			expectedLength: 5.0,
			expectedLenSq:  25.0,
		},
		{
			name:           "unit vector",
			vector:         func() linalg.Vector { return linalg.NewVector(1, 0, 0) },
			expectedLength: 1.0,
			expectedLenSq:  1.0,
		},
		{
			name:           "zero vector",
			vector:         func() linalg.Vector { return linalg.NewVector(0, 0, 0) },
			expectedLength: 0.0,
			expectedLenSq:  0.0,
		},
		{
			name:           "empty vector",
			vector:         func() linalg.Vector { return linalg.NewVector() },
			expectedLength: 0.0,
			expectedLenSq:  0.0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := tt.vector()
			assert.True(t, almostEqual(v.Length(), tt.expectedLength))
			assert.True(t, almostEqual(v.LengthSq(), tt.expectedLenSq))
		})
	}
}

// TestVectorNormalize tests the normalization method with table-driven tests.
func TestVectorNormalize(t *testing.T) {
	tests := []struct {
		name     string
		vector   func() linalg.Vector
		expected func() linalg.Vector
		wantErr  bool
	}{
		{
			name:     "3-4 vector normalization",
			vector:   func() linalg.Vector { return linalg.NewVector(3, 4) },
			expected: func() linalg.Vector { return linalg.NewVector(0.6, 0.8) },
			wantErr:  false,
		},
		{
			name:     "unit vector normalization",
			vector:   func() linalg.Vector { return linalg.NewVector(1, 0, 0) },
			expected: func() linalg.Vector { return linalg.NewVector(1, 0, 0) },
			wantErr:  false,
		},
		{
			name:    "zero vector normalization",
			vector:  func() linalg.Vector { return linalg.NewVector(0, 0) },
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := tt.vector()
			normalized, err := v.Normalize()
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testVectorsEqual(normalized, tt.expected()))
			// Verify that normalized vector has length 1
			assert.True(t, almostEqual(normalized.Length(), 1.0))
		})
	}
}

// TestVectorGetSet tests the Get and Set methods with table-driven tests.
func TestVectorGetSet(t *testing.T) {
	v := linalg.NewVector(1, 2, 3)

	getTests := []struct {
		name     string
		index    int
		expected float64
		wantErr  bool
	}{
		{"valid get index 0", 0, 1.0, false},
		{"valid get index 1", 1, 2.0, false},
		{"valid get index 2", 2, 3.0, false},
		{"negative index", -1, 0, true},
		{"index too large", 3, 0, true},
	}

	for _, tt := range getTests {
		t.Run(tt.name, func(t *testing.T) {
			val, err := v.Get(tt.index)
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
		index   int
		value   float64
		wantErr bool
	}{
		{"valid set index 0", 0, 99.0, false},
		{"valid set index 1", 1, -5.5, false},
		{"negative index", -1, 1.0, true},
		{"index too large", 3, 1.0, true},
	}

	for _, tt := range setTests {
		t.Run(tt.name, func(t *testing.T) {
			testV := linalg.NewVector(1, 2, 3) // Use a fresh vector for each test
			err := testV.Set(tt.index, tt.value)
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			val, _ := testV.Get(tt.index)
			assert.True(t, almostEqual(val, tt.value))
		})
	}
}

// TestNewVectorFromTensor tests the NewVectorFromTensor function with table-driven tests.
func TestNewVectorFromTensor(t *testing.T) {
	tests := []struct {
		name     string
		tensor   func() *linalg.Tensor
		expected func() linalg.Vector
		wantErr  bool
	}{
		{
			name: "valid 1D tensor conversion",
			tensor: func() *linalg.Tensor {
				tensor, _ := linalg.NewTensorWithData([]float64{1, 2, 3}, 3)
				return tensor
			},
			expected: func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
			wantErr:  false,
		},
		{
			name: "non-vector tensor (2D)",
			tensor: func() *linalg.Tensor {
				tensor, _ := linalg.NewTensorWithData([]float64{1, 2, 3, 4}, 2, 2)
				return tensor
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v, err := linalg.NewVectorFromTensor(tt.tensor())
			if tt.wantErr {
				assert.Error(t, err)
				return
			}
			assert.NoError(t, err)
			assert.True(t, testVectorsEqual(v, tt.expected()))
		})
	}
}

// TestVectorClone tests the Clone method with table-driven tests.
func TestVectorClone(t *testing.T) {
	tests := []struct {
		name   string
		vector func() linalg.Vector
	}{
		{
			name:   "3D vector clone",
			vector: func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
		},
		{
			name:   "empty vector clone",
			vector: func() linalg.Vector { return linalg.NewVector() },
		},
		{
			name:   "single element clone",
			vector: func() linalg.Vector { return linalg.NewVector(42.0) },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			original := tt.vector()
			cloned := original.Clone()

			// Check if clone has same data
			assert.True(t, testVectorsEqual(original, cloned))

			// Verify independence (only test if vector has elements)
			if original.Dim() > 0 {
				original.Set(0, 99.0)
				clonedVal, _ := cloned.Get(0)
				assert.False(t, almostEqual(clonedVal, 99.0))
			}
		})
	}
}

// TestVectorAsTensor tests the AsTensor method with table-driven tests.
func TestVectorAsTensor(t *testing.T) {
	tests := []struct {
		name     string
		vector   func() linalg.Vector
		expected []float64
		shape    []int
	}{
		{
			name:     "3D vector to tensor",
			vector:   func() linalg.Vector { return linalg.NewVector(1, 2, 3) },
			expected: []float64{1, 2, 3},
			shape:    []int{3},
		},
		{
			name:     "single element vector",
			vector:   func() linalg.Vector { return linalg.NewVector(42.0) },
			expected: []float64{42.0},
			shape:    []int{1},
		},
		{
			name:     "empty vector",
			vector:   func() linalg.Vector { return linalg.NewVector() },
			expected: []float64{},
			shape:    []int{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			v := tt.vector()
			tensor := v.AsTensor()

			// Check tensor properties
			if len(tt.shape) > 0 {
				assert.Equal(t, 1, tensor.Rank())
				assert.Equal(t, tt.shape, tensor.Shape())
			} else {
				assert.Equal(t, 0, tensor.Rank())
			}

			// Check data
			for i, expectedVal := range tt.expected {
				if len(tt.expected) > 0 {
					val, _ := tensor.Get(i)
					assert.True(t, almostEqual(val, expectedVal))
				}
			}

			// Verify independence (only test if vector has elements)
			if v.Dim() > 0 {
				tensor.Set(99.0, 0)
				originalVal, _ := v.Get(0)
				assert.False(t, almostEqual(originalVal, 99.0))
			}
		})
	}
}
