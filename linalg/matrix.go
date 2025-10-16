package linalg

import (
	"fmt"
	"math"
)

// Matrix は n x m の行列を表す汎用的な構造体です。
// データは行優先（Row-major）で1次元スライスに格納されます。
type Matrix struct {
	Data []float64 // 行優先で格納された要素
	Rows int       // 行数 (n)
	Cols int       // 列数 (m)
}

// NewMatrix は多次元スライス [][]float64 から新しい Matrix を作成します。
func NewMatrix(data [][]float64) (Matrix, error) {
	if len(data) == 0 {
		return Matrix{}, fmt.Errorf("入力データに行がありません")
	}

	rows := len(data)
	cols := len(data[0])
	totalSize := rows * cols

	flatData := make([]float64, 0, totalSize)

	for i, rowData := range data {
		if len(rowData) != cols {
			return Matrix{}, fmt.Errorf("行 %d の列数が異なります: 期待値 %d, 実際 %d", i, cols, len(rowData))
		}
		flatData = append(flatData, rowData...)
	}

	return Matrix{
		Data: flatData,
		Rows: rows,
		Cols: cols,
	}, nil
}

// Get は指定された i行 j列 の要素を返します。
func (m Matrix) Get(row, col int) (float64, error) {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		return 0, fmt.Errorf("インデックス (%d, %d) が行列の境界外です", row, col)
	}
	return m.Data[row*m.Cols+col], nil
}

// Set は指定された i行 j列 の要素を val で設定します。
func (m Matrix) Set(row, col int, val float64) error {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		return fmt.Errorf("インデックス (%d, %d) が行列の境界外です", row, col)
	}
	m.Data[row*m.Cols+col] = val
	return nil
}

// Add は現在の行列に別の行列を加算した新しい行列を返します。
func (m Matrix) Add(other Matrix) (Matrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return Matrix{}, fmt.Errorf("行列のサイズが異なります: %d x %d と %d x %d", m.Rows, m.Cols, other.Rows, other.Cols)
	}

	resultData := make([]float64, len(m.Data))
	for i := range m.Data {
		resultData[i] = m.Data[i] + other.Data[i]
	}

	return Matrix{Data: resultData, Rows: m.Rows, Cols: m.Cols}, nil
}

// Multiply は現在の行列（A）に別の行列（B）を乗算した新しい行列（C = A * B）を返します。
func (m Matrix) Multiply(other Matrix) (Matrix, error) {
	if m.Cols != other.Rows {
		return Matrix{}, fmt.Errorf("行列の乗算ルールに違反します: Aの列数(%d) と Bの行数(%d) が一致しません", m.Cols, other.Rows)
	}

	resultRows := m.Rows
	resultCols := other.Cols
	resultData := make([]float64, resultRows*resultCols)

	for i := 0; i < resultRows; i++ {
		for j := 0; j < resultCols; j++ {
			var sum float64
			for k := 0; k < m.Cols; k++ {
				sum += m.Data[i*m.Cols+k] * other.Data[k*other.Cols+j]
			}
			resultData[i*resultCols+j] = sum
		}
	}

	return Matrix{Data: resultData, Rows: resultRows, Cols: resultCols}, nil
}

// Transpose は現在の行列の転置行列を返します。
func (m Matrix) Transpose() Matrix {
	resultData := make([]float64, m.Rows*m.Cols)

	// A[i][j] は B[j][i] になる (B.Rows = A.Cols, B.Cols = A.Rows)
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			originalIndex := i*m.Cols + j
			transposedIndex := j*m.Rows + i // 新しい行列の行数は m.Cols
			resultData[transposedIndex] = m.Data[originalIndex]
		}
	}

	return Matrix{
		Data: resultData,
		Rows: m.Cols, // 行数と列数が入れ替わる
		Cols: m.Rows,
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
		return m.Data[0], nil
	}
	if n == 2 {
		// ad - bc
		return m.Data[0]*m.Data[3] - m.Data[1]*m.Data[2], nil
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

// Inverse はガウス・ジョルダン法を使用して現在の行列の逆行列を計算します。（コードは省略せず全て記載）
func (m Matrix) Inverse() (Matrix, error) {
	if m.Rows != m.Cols {
		return Matrix{}, fmt.Errorf("逆行列は正方行列（%d x %d）にのみ定義されます", m.Rows, m.Cols)
	}

	n := m.Rows
	augmentedData := make([]float64, n*2*n)
	augmentedCols := n * 2

	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			augmentedData[i*augmentedCols+j] = m.Data[i*n+j]
			if i == j {
				augmentedData[i*augmentedCols+n+j] = 1.0
			} else {
				augmentedData[i*augmentedCols+n+j] = 0.0
			}
		}
	}

	augmentedMatrix := Matrix{Data: augmentedData, Rows: n, Cols: augmentedCols}
	const epsilon = 1e-9

	for i := 0; i < n; i++ {
		pivotRow := i
		for k := i + 1; k < n; k++ {
			if math.Abs(augmentedMatrix.Data[k*augmentedCols+i]) > math.Abs(augmentedMatrix.Data[pivotRow*augmentedCols+i]) {
				pivotRow = k
			}
		}

		if math.Abs(augmentedMatrix.Data[pivotRow*augmentedCols+i]) < epsilon {
			return Matrix{}, fmt.Errorf("行列は特異です（行列式がゼロに近い）：逆行列が存在しません")
		}

		if pivotRow != i {
			for j := 0; j < augmentedCols; j++ {
				idx_i := i*augmentedCols + j
				idx_pivot := pivotRow*augmentedCols + j
				augmentedMatrix.Data[idx_i], augmentedMatrix.Data[idx_pivot] = augmentedMatrix.Data[idx_pivot], augmentedMatrix.Data[idx_i]
			}
		}

		pivotVal := augmentedMatrix.Data[i*augmentedCols+i]
		for j := i; j < augmentedCols; j++ {
			augmentedMatrix.Data[i*augmentedCols+j] /= pivotVal
		}

		for k := 0; k < n; k++ {
			if k != i {
				factor := augmentedMatrix.Data[k*augmentedCols+i]
				for j := i; j < augmentedCols; j++ {
					idx_k := k*augmentedCols + j
					idx_i := i*augmentedCols + j
					augmentedMatrix.Data[idx_k] -= factor * augmentedMatrix.Data[idx_i]
				}
			}
		}
	}

	inverseData := make([]float64, n*n)
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			inverseData[i*n+j] = augmentedMatrix.Data[i*augmentedCols+n+j]
		}
	}

	return Matrix{Data: inverseData, Rows: n, Cols: n}, nil
}

// IsDiagonal は行列が対角行列（非対角要素が全てゼロ）であるかをチェックします。
func (m Matrix) IsDiagonal() bool {
	if m.Rows != m.Cols {
		return false
	} // 対角行列は正方行列である必要がある

	const epsilon = 1e-9
	for i := 0; i < m.Rows; i++ {
		for j := 0; j < m.Cols; j++ {
			if i != j {
				if math.Abs(m.Data[i*m.Cols+j]) > epsilon {
					return false
				}
			}
		}
	}
	return true
}

// IsSymmetric は行列が対称行列（A = A^T）であるかをチェックします。
func (m Matrix) IsSymmetric() bool {
	if m.Rows != m.Cols {
		return false
	}

	const epsilon = 1e-9
	for i := 0; i < m.Rows; i++ {
		for j := i + 1; j < m.Cols; j++ { // 対角要素より上の部分のみチェック
			val_ij := m.Data[i*m.Cols+j]
			val_ji := m.Data[j*m.Cols+i]
			if math.Abs(val_ij-val_ji) > epsilon {
				return false
			}
		}
	}
	return true
}

// IsUpperTriangular は行列が上三角行列（対角要素より下の要素が全てゼロ）であるかをチェックします。
func (m Matrix) IsUpperTriangular() bool {
	if m.Rows != m.Cols {
		return false
	}

	const epsilon = 1e-9
	for i := 1; i < m.Rows; i++ { // 1行目から
		for j := 0; j < i; j++ { // 対角要素より左下の要素をチェック
			if math.Abs(m.Data[i*m.Cols+j]) > epsilon {
				return false
			}
		}
	}
	return true
}

// IsLowerTriangular は行列が下三角行列（対角要素より上の要素が全てゼロ）であるかをチェックします。
func (m Matrix) IsLowerTriangular() bool {
	if m.Rows != m.Cols {
		return false
	}

	const epsilon = 1e-9
	for i := 0; i < m.Rows; i++ {
		for j := i + 1; j < m.Cols; j++ { // 対角要素より右上の要素をチェック
			if math.Abs(m.Data[i*m.Cols+j]) > epsilon {
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
