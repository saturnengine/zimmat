package linalg_test

import (
	"testing"
	"github.com/saturnengine/zimmat/linalg" 
)

// testMatricesEqual は2つの行列が等しいか（要素、行、列が許容誤差内で一致するか）をチェックします。
func testMatricesEqual(m1, m2 linalg.Matrix) bool {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return false
	}
	if len(m1.Data) != len(m2.Data) {
		return false
	}

	for i := range m1.Data {
		if !almostEqual(m1.Data[i], m2.Data[i]) {
			return false
		}
	}
	return true
}

// TestNewMatrix は NewMatrix 関数のテストです。
func TestNewMatrix(t *testing.T) {
	// 正常な初期化 (2x3)
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
	}
	m, err := linalg.NewMatrix(data)
	if err != nil {
		t.Fatalf("NewMatrixでエラーが発生しました: %v", err)
	}
	if m.Rows != 2 || m.Cols != 3 {
		t.Errorf("サイズが期待値と異なります。期待値: 2x3, 実際: %dx%d", m.Rows, m.Cols)
	}
}

// TestMatrixGetAndSet は Get と Set メソッドのテストです。
func TestMatrixGetAndSet(t *testing.T) { 
	data := [][]float64{{1.1, 2.2}, {3.3, 4.4}}
	m, _ := linalg.NewMatrix(data)

	// Getのテスト
	val, _ := m.Get(1, 0)
	if !almostEqual(val, 3.3) {
		t.Errorf("Getの結果が期待値と異なります。期待値: 3.3, 実際: %f", val)
	}

	// Setのテスト
	m.Set(0, 1, 9.9)
	val, _ = m.Get(0, 1)
	if !almostEqual(val, 9.9) {
		t.Errorf("Setの結果が反映されていません。期待値: 9.9, 実際: %f", val)
	}
}

// TestMatrixAdd は行列の加算メソッドのテストです。
func TestMatrixAdd(t *testing.T) { 
	m1, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	m2, _ := linalg.NewMatrix([][]float64{{5, 6}, {7, 8}})
	expected, _ := linalg.NewMatrix([][]float64{{6, 8}, {10, 12}})

	result, _ := m1.Add(m2)
	if !testMatricesEqual(result, expected) {
		t.Errorf("Addの結果が期待値と異なります。期待値: %v\n実際: %v", expected.Data, result.Data)
	}
}

// TestMatrixMultiply は行列の乗算メソッドのテストです。
func TestMatrixMultiply(t *testing.T) { 
	// A (2x3) * B (3x2) = C (2x2)
	m_A, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}}) 
	m_B, _ := linalg.NewMatrix([][]float64{{7, 8}, {9, 10}, {11, 12}}) 
	expected, _ := linalg.NewMatrix([][]float64{{58, 64}, {139, 154}}) 

	result, _ := m_A.Multiply(m_B)
	if !testMatricesEqual(result, expected) {
		t.Errorf("Multiplyの結果が期待値と異なります。\n期待値: %v\n実際: %v", expected.Data, result.Data)
	}
}

// TestMatrixTranspose は転置行列メソッドのテストです。
func TestMatrixTranspose(t *testing.T) {
	// 2x3 行列
	m, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}})
	// 期待される 3x2 行列
	expected, _ := linalg.NewMatrix([][]float64{{1, 4}, {2, 5}, {3, 6}})
	
	result := m.Transpose()
	
	if !testMatricesEqual(result, expected) {
		t.Errorf("Transposeの結果が期待値と異なります。\n期待値: %v\n実際: %v", expected.Data, result.Data)
	}
	if result.Rows != 3 || result.Cols != 2 {
		t.Errorf("転置後のサイズが誤っています。期待値: 3x2, 実際: %dx%d", result.Rows, result.Cols)
	}
}

// TestMatrixDeterminant は行列式メソッドのテストです。
func TestMatrixDeterminant(t *testing.T) {
	// 2x2: det(A) = 4*6 - 7*2 = 10
	m2x2, _ := linalg.NewMatrix([][]float64{{4, 7}, {2, 6}})
	det2x2, _ := m2x2.Determinant()
	if !almostEqual(det2x2, 10.0) {
		t.Errorf("2x2 行列式の結果が誤っています。期待値: 10.0, 実際: %f", det2x2)
	}

	// 3x3: det(A) = 27
	m3x3, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 0}})
	det3x3, _ := m3x3.Determinant()
	if !almostEqual(det3x3, 27.0) {
		t.Errorf("3x3 行列式の結果が誤っています。期待値: 27.0, 実際: %f", det3x3)
	}
	
	// 非正方行列のテスト
	mNonSquare, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}})
	_, err := mNonSquare.Determinant()
	if err == nil {
		t.Error("非正方行列に対してエラーが返されませんでした")
	}
}

// TestMatrixInverse は逆行列計算メソッドのテストです。
func TestMatrixInverse(t *testing.T) {
	// 正常な 2x2 行列 (det=10)
	m2x2, _ := linalg.NewMatrix([][]float64{{4, 7}, {2, 6}})
	expected2x2, _ := linalg.NewMatrix([][]float64{{0.6, -0.7}, {-0.2, 0.4}})

	inv2x2, _ := m2x2.Inverse()
	if !testMatricesEqual(inv2x2, expected2x2) {
		t.Errorf("2x2 Inverseの結果が期待値と異なります。\n期待値: %v\n実際: %v", expected2x2.Data, inv2x2.Data)
	}

    // 検算: A * A_inv が単位行列 I になるか確認
    product, _ := m2x2.Multiply(inv2x2)
    identity2x2, _ := linalg.NewMatrix([][]float64{{1.0, 0.0}, {0.0, 1.0}})
    if !testMatricesEqual(product, identity2x2) {
        t.Errorf("2x2 Inverseの検算 (A * A_inv) が失敗しました。結果: %v", product.Data)
    }

	// 特異行列（逆行列が存在しない）のテスト
	singularM, _ := linalg.NewMatrix([][]float64{{2, 4}, {1, 2}})
	_, err := singularM.Inverse()
	if err == nil {
		t.Error("特異行列に対してエラーが返されませんでした")
	}
}

// TestMatrixSpecialTypes は対角、対称、三角行列の判定メソッドのテストです。
func TestMatrixSpecialTypes(t *testing.T) {
    // 1. 対角行列
    diagM, _ := linalg.NewMatrix([][]float64{{1, 0}, {0, 2}})
    nonDiagM, _ := linalg.NewMatrix([][]float64{{1, 1}, {0, 2}})
    if !diagM.IsDiagonal() { t.Error("IsDiagonal: 対角行列を正しく判定できませんでした") }
    if nonDiagM.IsDiagonal() { t.Error("IsDiagonal: 非対角行列を誤って判定しました") }
    
    // 2. 対称行列
    symM, _ := linalg.NewMatrix([][]float64{{1, 2}, {2, 3}})
    nonSymM, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
    if !symM.IsSymmetric() { t.Error("IsSymmetric: 対称行列を正しく判定できませんでした") }
    if nonSymM.IsSymmetric() { t.Error("IsSymmetric: 非対称行列を誤って判定しました") }
    
    // 3. 上三角行列
    upperM, _ := linalg.NewMatrix([][]float64{{1, 2}, {0, 3}})
    if !upperM.IsUpperTriangular() { t.Errorf("IsUpperTriangular: 上三角行列を正しく判定できませんでした") }
    if upperM.IsLowerTriangular() { t.Errorf("IsLowerTriangular: 上三角行列を誤って判定しました") }
    
    // 4. 下三角行列
    lowerM, _ := linalg.NewMatrix([][]float64{{1, 0}, {2, 3}})
    if !lowerM.IsLowerTriangular() { t.Errorf("IsLowerTriangular: 下三角行列を正しく判定できませんでした") }
    if lowerM.IsUpperTriangular() { t.Errorf("IsUpperTriangular: 下三角行列を誤って判定しました") }
}