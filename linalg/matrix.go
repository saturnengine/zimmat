package linalg

import (
	"fmt"
)

// Matrix は n x m の行列を表す汎用的な構造体です。
// データは行優先（Row-major）で1次元スライスに格納されます。
type Matrix struct {
	Data []float64 // 行優先で格納された要素
	Rows int       // 行数 (n)
	Cols int       // 列数 (m)
}

// NewMatrix は指定された行数、列数、要素から新しい Matrix を作成します。
// 要素の数（len(elements)）は Rows * Cols と一致する必要があります。
func NewMatrix(rows, cols int, elements ...float64) (Matrix, error) {
	if rows <= 0 || cols <= 0 {
		return Matrix{}, fmt.Errorf("行数と列数は1以上である必要があります")
	}
	if len(elements) != rows*cols {
		return Matrix{}, fmt.Errorf("指定された要素数（%d）は、行列のサイズ（%d x %d = %d）と一致しません", len(elements), rows, cols, rows*cols)
	}

	return Matrix{
		Data: elements,
		Rows: rows,
		Cols: cols,
	}, nil
}

// Get は指定された i行 j列 の要素を返します。
// インデックスは0から始まります。
func (m Matrix) Get(row, col int) (float64, error) {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		return 0, fmt.Errorf("インデックス (%d, %d) が行列の境界外です", row, col)
	}
	// 行優先のインデックス計算: index = row * Cols + col
	return m.Data[row*m.Cols+col], nil
}

// Set は指定された i行 j列 の要素を val で設定します。
func (m Matrix) Set(row, col int, val float64) error {
	if row < 0 || row >= m.Rows || col < 0 || col >= m.Cols {
		return fmt.Errorf("インデックス (%d, %d) が行列の境界外です", row, col)
	}
	// 行優先のインデックス計算: index = row * Cols + col
	m.Data[row*m.Cols+col] = val
	return nil
}

// Add は現在の行列に別の行列を加算した新しい行列を返します。
// サイズが一致しない場合はエラーを返します。
func (m Matrix) Add(other Matrix) (Matrix, error) {
	if m.Rows != other.Rows || m.Cols != other.Cols {
		return Matrix{}, fmt.Errorf("行列のサイズが異なります: %d x %d と %d x %d", m.Rows, m.Cols, other.Rows, other.Cols)
	}

	resultData := make([]float64, len(m.Data))
	for i := range m.Data {
		resultData[i] = m.Data[i] + other.Data[i]
	}

	// エラーは発生しないことが保証される
	result, _ := NewMatrix(m.Rows, m.Cols, resultData...)
	return result, nil
}

// Multiply は現在の行列（A）に別の行列（B）を乗算した新しい行列（C = A * B）を返します。
// m.Cols (Aの列数) と other.Rows (Bの行数) が一致しない場合はエラーを返します。
func (m Matrix) Multiply(other Matrix) (Matrix, error) {
	if m.Cols != other.Rows {
		return Matrix{}, fmt.Errorf("行列の乗算ルールに違反します: Aの列数(%d) と Bの行数(%d) が一致しません", m.Cols, other.Rows)
	}

	resultRows := m.Rows
	resultCols := other.Cols
	resultData := make([]float64, resultRows*resultCols)

	// 行列乗算: C[i][j] = Σ(k=0 to m.Cols-1) A[i][k] * B[k][j]
	for i := 0; i < resultRows; i++ { // Cの行
		for j := 0; j < resultCols; j++ { // Cの列
			var sum float64
			for k := 0; k < m.Cols; k++ { // 共通の次元
				// 行優先インデックス: A[i][k] = i*m.Cols + k
				// 行優先インデックス: B[k][j] = k*other.Cols + j
				sum += m.Data[i*m.Cols+k] * other.Data[k*other.Cols+j]
			}
			// 行優先インデックス: C[i][j] = i*resultCols + j
			resultData[i*resultCols+j] = sum
		}
	}

	result, _ := NewMatrix(resultRows, resultCols, resultData...)
	return result, nil
}