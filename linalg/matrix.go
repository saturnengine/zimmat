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
	tensor *Tensor    // Internal tensor (2-dimensional)
	Data   []float64  // Public field for compatibility (reference to tensor's Data)
	Rows   int        // Number of rows (n)
	Cols   int        // Number of columns (m)
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
func NewMatrix(data [][]float64) (Matrix, error) {
	if len(data) == 0 {
		return Matrix{}, fmt.Errorf("入力データに行がありません")
	}

	rows := len(data)
	cols := len(data[0])
	
	flatData := make([]float64, 0, rows*cols)

	for i, rowData := range data {
		if len(rowData) != cols {
			return Matrix{}, fmt.Errorf("行 %d の列数が異なります: 期待値 %d, 実際 %d", i, cols, len(rowData))
		}
		flatData = append(flatData, rowData...)
	}

	tensor, err := NewTensorWithData(flatData, rows, cols)
	if err != nil {
		return Matrix{}, fmt.Errorf("行列作成エラー: %v", err)
	}

	return Matrix{
		tensor: tensor,
		Data:   tensor.Data,
		Rows:   rows,
		Cols:   cols,
	}, nil
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
func NewMatrixFromTensor(tensor *Tensor) (Matrix, error) {
	if !tensor.IsMatrix() {
		return Matrix{}, fmt.Errorf("テンソル（形状: %v）は行列ではありません", tensor.Shape)
	}
	
	return Matrix{
		tensor: tensor.Clone(),
		Data:   tensor.Data,
		Rows:   tensor.Shape[0],
		Cols:   tensor.Shape[1],
	}, nil
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
func NewZeroMatrix(rows, cols int) Matrix {
	tensor := NewTensor(rows, cols)
	return Matrix{
		tensor: tensor,
		Data:   tensor.Data,
		Rows:   rows,
		Cols:   cols,
	}
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
func NewIdentityMatrix(size int) Matrix {
	m := NewZeroMatrix(size, size)
	for i := 0; i < size; i++ {
		m.Set(i, i, 1.0)
	}
	return m
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
func (m Matrix) Get(row, col int) (float64, error) {
	return m.tensor.Get(row, col)
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
func (m Matrix) Set(row, col int, val float64) error {
	err := m.tensor.Set(val, row, col)
	if err != nil {
		return err
	}
	// Dataフィールドも更新（参照なので自動的に更新されるが、明示的に保証）
	m.Data = m.tensor.Data
	return nil
}

// Add は現在の行列に別の行列を加算した新しい行列を返します。
func (m Matrix) Add(other Matrix) (Matrix, error) {
	result, err := m.tensor.Add(other.tensor)
	if err != nil {
		return Matrix{}, fmt.Errorf("行列加算エラー: %v", err)
	}
	
	matrixResult, err := NewMatrixFromTensor(result)
	if err != nil {
		return Matrix{}, fmt.Errorf("結果行列作成エラー: %v", err)
	}
	
	return matrixResult, nil
}

// Subtract は現在の行列から別の行列を減算した新しい行列を返します。
func (m Matrix) Subtract(other Matrix) (Matrix, error) {
	result, err := m.tensor.Subtract(other.tensor)
	if err != nil {
		return Matrix{}, fmt.Errorf("行列減算エラー: %v", err)
	}
	
	matrixResult, err := NewMatrixFromTensor(result)
	if err != nil {
		return Matrix{}, fmt.Errorf("結果行列作成エラー: %v", err)
	}
	
	return matrixResult, nil
}

// Scale は行列をスカラー値で乗算した新しい行列を返します。
func (m Matrix) Scale(scalar float64) Matrix {
	result := m.tensor.Scale(scalar)
	
	matrixResult, err := NewMatrixFromTensor(result)
	if err != nil {
		panic(fmt.Sprintf("スケール結果行列作成エラー: %v", err))
	}
	
	return matrixResult
}

// Multiply は現在の行列（A）に別の行列（B）を乗算した新しい行列（C = A * B）を返します。
func (m Matrix) Multiply(other Matrix) (Matrix, error) {
	result, err := m.tensor.MatrixMultiply(other.tensor)
	if err != nil {
		return Matrix{}, fmt.Errorf("行列乗算エラー: %v", err)
	}
	
	matrixResult, err := NewMatrixFromTensor(result)
	if err != nil {
		return Matrix{}, fmt.Errorf("結果行列作成エラー: %v", err)
	}
	
	return matrixResult, nil
}

// Transpose は現在の行列の転置行列を返します。
func (m Matrix) Transpose() Matrix {
	result, err := m.tensor.Transpose()
	if err != nil {
		panic(fmt.Sprintf("行列転置エラー: %v", err))
	}
	
	matrixResult, err := NewMatrixFromTensor(result)
	if err != nil {
		panic(fmt.Sprintf("転置結果行列作成エラー: %v", err))
	}
	
	return matrixResult
}

// AsTensor はMatrixをTensorとして取得します。
func (m Matrix) AsTensor() *Tensor {
	return m.tensor.Clone()
}

// Clone は行列の完全なコピーを作成します。
func (m Matrix) Clone() Matrix {
	clonedTensor := m.tensor.Clone()
	return Matrix{
		tensor: clonedTensor,
		Data:   clonedTensor.Data,
		Rows:   m.Rows,
		Cols:   m.Cols,
	}
}

// Determinant は正方行列の行列式を計算します。
// 3x3まではサラスの公式/余因子展開、それ以上はLU分解などに頼るのが一般的ですが、
// ここでは余因子展開（再帰）を単純実装します。
func (m Matrix) Determinant() (float64, error) {
	if m.Rows != m.Cols {
		return 0, fmt.Errorf("行列式は正方行列（%d x %d）にのみ定義されます", m.Rows, m.Cols)
	}
	n := m.Rows

	if n == 1 {
		val, _ := m.Get(0, 0)
		return val, nil
	}
	if n == 2 {
		// ad - bc
		a, _ := m.Get(0, 0)
		b, _ := m.Get(0, 1)
		c, _ := m.Get(1, 0)
		d, _ := m.Get(1, 1)
		return a*d - b*c, nil
	}
    
    // 3x3以上は再帰的な余因子展開を使用（計算コストは高い）
	var det float64
	for j := 0; j < n; j++ {
		// i=0 の行について余因子展開
		sign := 1.0
		if j%2 != 0 {
			sign = -1.0
		}
        
        // Minor行列の作成
        minorData := make([][]float64, n-1)
        for row := 1; row < n; row++ { // 0行目を除外
            minorRow := make([]float64, 0, n-1)
            for col := 0; col < n; col++ {
                if col != j { // j列目を除外
                    val, _ := m.Get(row, col)
                    minorRow = append(minorRow, val)
                }
            }
            minorData[row-1] = minorRow
        }
        
        minorM, _ := NewMatrix(minorData) // Minor行列
        minorDet, _ := minorM.Determinant()
        
        a0j, _ := m.Get(0, j)
		det += sign * a0j * minorDet
	}

	return det, nil
}

// Inverse はガウス・ジョルダン法を使用して現在の行列の逆行列を計算します。
func (m Matrix) Inverse() (Matrix, error) {
	if m.Rows != m.Cols {
		return Matrix{}, fmt.Errorf("逆行列は正方行列（%d x %d）にのみ定義されます", m.Rows, m.Cols)
	}

	n := m.Rows
	// 拡張行列 [A|I] を作成
	augmented := NewZeroMatrix(n, 2*n)
	
	// 元の行列を左側にコピー
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			val, _ := m.Get(i, j)
			augmented.Set(i, j, val)
		}
	}
	
	// 右側に単位行列をセット
	for i := 0; i < n; i++ {
		augmented.Set(i, n+i, 1.0)
	}

	const epsilon = 1e-9

	// ガウス・ジョルダン消去法
	for i := 0; i < n; i++ {
		// ピボット選択
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
			return Matrix{}, fmt.Errorf("行列は特異です（行列式がゼロに近い）：逆行列が存在しません")
		}

		// 行交換
		if pivotRow != i {
			for j := 0; j < 2*n; j++ {
				val1, _ := augmented.Get(i, j)
				val2, _ := augmented.Get(pivotRow, j)
				augmented.Set(i, j, val2)
				augmented.Set(pivotRow, j, val1)
			}
		}

		// ピボット行を正規化
		pivotVal, _ = augmented.Get(i, i)
		for j := i; j < 2*n; j++ {
			val, _ := augmented.Get(i, j)
			augmented.Set(i, j, val/pivotVal)
		}

		// 他の行を消去
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

	// 右側の部分（逆行列）を抽出
	result := NewZeroMatrix(n, n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			val, _ := augmented.Get(i, n+j)
			result.Set(i, j, val)
		}
	}
    
    return result, nil
}

// IsDiagonal は行列が対角行列（非対角要素が全てゼロ）であるかをチェックします。
func (m Matrix) IsDiagonal() bool {
    if m.Rows != m.Cols { return false } // 対角行列は正方行列である必要がある
    
    const epsilon = 1e-9
    for i := 0; i < m.Rows; i++ {
        for j := 0; j < m.Cols; j++ {
            if i != j {
                val, _ := m.Get(i, j)
                if math.Abs(val) > epsilon {
                    return false
                }
            }
        }
    }
    return true
}

// IsSymmetric は行列が対称行列（A = A^T）であるかをチェックします。
func (m Matrix) IsSymmetric() bool {
    if m.Rows != m.Cols { return false }
    
    const epsilon = 1e-9
    for i := 0; i < m.Rows; i++ {
        for j := i + 1; j < m.Cols; j++ { // 対角要素より上の部分のみチェック
            val_ij, _ := m.Get(i, j)
            val_ji, _ := m.Get(j, i)
            if math.Abs(val_ij - val_ji) > epsilon {
                return false
            }
        }
    }
    return true
}

// IsUpperTriangular は行列が上三角行列（対角要素より下の要素が全てゼロ）であるかをチェックします。
func (m Matrix) IsUpperTriangular() bool {
    if m.Rows != m.Cols { return false }
    
    const epsilon = 1e-9
    for i := 1; i < m.Rows; i++ { // 1行目から
        for j := 0; j < i; j++ { // 対角要素より左下の要素をチェック
            val, _ := m.Get(i, j)
            if math.Abs(val) > epsilon {
                return false
            }
        }
    }
    return true
}

// IsLowerTriangular は行列が下三角行列（対角要素より上の要素が全てゼロ）であるかをチェックします。
func (m Matrix) IsLowerTriangular() bool {
    if m.Rows != m.Cols { return false }
    
    const epsilon = 1e-9
    for i := 0; i < m.Rows; i++ {
        for j := i + 1; j < m.Cols; j++ { // 対角要素より右上の要素をチェック
            val, _ := m.Get(i, j)
            if math.Abs(val) > epsilon {
                return false
            }
        }
    }
    return true
}

/*
// --- 高度な線形代数機能のスケルトン (実装は省略) ---
*/

// PseudoInverse はムーア・ペンローズ型擬似逆行列 (A+) を計算します。
// これは一般にSVD (特異値分解) を用いて計算されます。
/*
func (m Matrix) PseudoInverse() (Matrix, error) {
    // 実際には SVD (Singular Value Decomposition) の実装が必要
    return Matrix{}, fmt.Errorf("擬似逆行列の計算は未実装です（SVDが必要です）")
}
*/

// EigenSystem は固有値と固有ベクトルを計算します。
/*
func (m Matrix) EigenSystem() ([]complex128, []Vector, error) {
    // 実際には QR法やJacobi法などの反復アルゴリズムの実装が必要
    return nil, nil, fmt.Errorf("固有値と固有ベクトルの計算は未実装です")
}
*/