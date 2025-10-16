package linalg_test

import (
	"testing"
	"github.com/saturnengine/zimmat/linalg" 
)

// almostEqual, floatTolerance, testVectorsEqual は vector_test.go で定義されているが、
// matrix_test.go でも必要なので、ここでは testMatricesEqual のみ再掲し、
// 前提として almostEqual 等は存在するものとします。

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
	m, err := linalg.NewMatrix(data) // ★ 多次元スライスで呼び出し
	if err != nil {
		t.Fatalf("NewMatrixでエラーが発生しました: %v", err)
	}
	if m.Rows != 2 || m.Cols != 3 {
		t.Errorf("サイズが期待値と異なります。期待値: 2x3, 実際: %dx%d", m.Rows, m.Cols)
	}
    
    // 内部のフラットデータを確認
    expectedFlatData := []float64{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}
    for i := range expectedFlatData {
        if !almostEqual(m.Data[i], expectedFlatData[i]) {
            t.Errorf("内部データが期待値と異なります。期待値: %v, 実際: %v", expectedFlatData, m.Data)
            break
        }
    }


	// 不正な形状のエラーテスト (行によって列数が異なる)
	badData := [][]float64{
		{1, 2, 3},
		{4, 5}, // 3列必要だが2列しかない
	}
	_, err = linalg.NewMatrix(badData)
	if err == nil {
		t.Error("不正な形状の場合にエラーが返されませんでした")
	}

	// 空の入力のエラーテスト
	_, err = linalg.NewMatrix([][]float64{})
	if err == nil {
		t.Error("空の入力でエラーが返されませんでした")
	}
}

// TestMatrixGetAndSet は Get と Set メソッドのテストです。
func TestMatrixGetAndSet(t *testing.T) { // 名前を修正
	data := [][]float64{{1.1, 2.2}, {3.3, 4.4}} // ★ 多次元スライス
	m, _ := linalg.NewMatrix(data)

	// Getのテスト
	val, err := m.Get(1, 0)
	if err != nil {
		t.Fatalf("Getでエラーが発生しました: %v", err)
	}
	if !almostEqual(val, 3.3) {
		t.Errorf("Getの結果が期待値と異なります。期待値: 3.3, 実際: %f", val)
	}

	// Setのテスト
	m.Set(0, 1, 9.9)
	val, _ = m.Get(0, 1)
	if !almostEqual(val, 9.9) {
		t.Errorf("Setの結果が反映されていません。期待値: 9.9, 実際: %f", val)
	}

	// 境界外アクセス（Get）のテスト
	_, err = m.Get(2, 0)
	if err == nil {
		t.Error("境界外Getでエラーが返されませんでした")
	}

	// 境界外アクセス（Set）のテスト
	err = m.Set(0, 2, 0)
	if err == nil {
		t.Error("境界外Setでエラーが返されませんでした")
	}
}

// TestMatrixAdd は行列の加算メソッドのテストです。
func TestMatrixAdd(t *testing.T) { // 名前を修正
	m1, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})        // ★ 修正
	m2, _ := linalg.NewMatrix([][]float64{{5, 6}, {7, 8}})        // ★ 修正
	expected, _ := linalg.NewMatrix([][]float64{{6, 8}, {10, 12}}) // ★ 修正

	result, err := m1.Add(m2)
	if err != nil {
		t.Fatalf("Addでエラーが発生しました: %v", err)
	}

	if !testMatricesEqual(result, expected) {
		t.Errorf("Addの結果が期待値と異なります。\n期待値: %v\n実際: %v", expected.Data, result.Data)
	}

	// サイズ不一致のテスト
	m3, _ := linalg.NewMatrix([][]float64{{1, 2, 3}, {4, 5, 6}})
	_, err = m1.Add(m3)
	if err == nil {
		t.Error("サイズ不一致の場合にエラーが返されませんでした")
	}
}

// TestMatrixMultiply は行列の乗算メソッドのテストです。
func TestMatrixMultiply(t *testing.T) { // 名前を修正
	// A (2x3)
	m_A, _ := linalg.NewMatrix([][]float64{
		{1, 2, 3}, 
		{4, 5, 6},
	}) 
    
	// B (3x2)
	m_B, _ := linalg.NewMatrix([][]float64{
		{7, 8}, 
		{9, 10}, 
		{11, 12},
	}) 

	// 期待される結果 C (2x2)
	expected, _ := linalg.NewMatrix([][]float64{
		{58, 64}, 
		{139, 154},
	}) 

	result, err := m_A.Multiply(m_B)
	if err != nil {
		t.Fatalf("Multiplyでエラーが発生しました: %v", err)
	}

	if !testMatricesEqual(result, expected) {
		t.Errorf("Multiplyの結果が期待値と異なります。\n期待値: %v\n実際: %v", expected.Data, result.Data)
	}

	// 乗算ルール違反のテスト
	m_C, _ := linalg.NewMatrix([][]float64{{1, 2}, {3, 4}})
	_, err = m_A.Multiply(m_C) // A(2x3) * C(2x2) -> 3 != 2 でエラー
	if err == nil {
		t.Error("乗算ルール違反の場合にエラーが返されませんでした")
	}
}