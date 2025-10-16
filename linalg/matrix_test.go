package linalg_test

import (
	"testing"
	"github.com/saturnengine/zimmat/linalg" // モジュール名に合わせてインポートパスを修正
)

// testMatricesEqual は2つの行列が等しいか（要素、行、列が許容誤差内で一致するか）をチェックします。
func testMatricesEqual(m1, m2 linalg.Matrix) bool {
	if m1.Rows != m2.Rows || m1.Cols != m2.Cols {
		return false
	}
	if len(m1.Data) != len(m2.Data) {
		return false // サイズが一致すればDataの長さも一致するはずだが、念のため
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
	m, err := linalg.NewMatrix(2, 3, 1, 2, 3, 4, 5, 6)
	if err != nil {
		t.Fatalf("NewMatrixでエラーが発生しました: %v", err)
	}
	if m.Rows != 2 || m.Cols != 3 {
		t.Errorf("サイズが期待値と異なります。期待値: 2x3, 実際: %dx%d", m.Rows, m.Cols)
	}

	// 要素数不一致のエラーテスト
	_, err = linalg.NewMatrix(2, 3, 1, 2, 3, 4, 5) // 6要素必要だが5つ
	if err == nil {
		t.Error("要素数不一致の場合にエラーが返されませんでした")
	}

	// 無効なサイズのエラーテスト
	_, err = linalg.NewMatrix(0, 3, 1, 2, 3)
	if err == nil {
		t.Error("無効な行数でエラーが返されませんでした")
	}
}

// TestMatrixGetAndSet は Get と Set メソッドのテストです。
func TestMatrixGetAndSet(t *testing.T) {
	data := []float64{1.1, 2.2, 3.3, 4.4}
	m, _ := linalg.NewMatrix(2, 2, data...)

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
func TestMatrixAdd(t *testing.T) {
	m1, _ := linalg.NewMatrix(2, 2, 1, 2, 3, 4)
	m2, _ := linalg.NewMatrix(2, 2, 5, 6, 7, 8)
	expected, _ := linalg.NewMatrix(2, 2, 6, 8, 10, 12)

	result, err := m1.Add(m2)
	if err != nil {
		t.Fatalf("Addでエラーが発生しました: %v", err)
	}

	if !testMatricesEqual(result, expected) {
		t.Errorf("Addの結果が期待値と異なります。\n期待値: %v\n実際: %v", expected.Data, result.Data)
	}

	// サイズ不一致のテスト
	m3, _ := linalg.NewMatrix(2, 3, 1, 2, 3, 4, 5, 6)
	_, err = m1.Add(m3)
	if err == nil {
		t.Error("サイズ不一致の場合にエラーが返されませんでした")
	}
}

// TestMatrixMultiply は行列の乗算メソッドのテストです。
func TestMatrixMultiply(t *testing.T) {
	// A (2x3) * B (3x2) = C (2x2)
	m_A, _ := linalg.NewMatrix(2, 3, 1, 2, 3, 4, 5, 6)
	m_B, _ := linalg.NewMatrix(3, 2, 7, 8, 9, 10, 11, 12)
	// C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
	// C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
	// ...
	expected, _ := linalg.NewMatrix(2, 2, 58, 64, 139, 154)

	result, err := m_A.Multiply(m_B)
	if err != nil {
		t.Fatalf("Multiplyでエラーが発生しました: %v", err)
	}

	if !testMatricesEqual(result, expected) {
		t.Errorf("Multiplyの結果が期待値と異なります。\n期待値: %v\n実際: %v", expected.Data, result.Data)
	}

	// 乗算ルール違反のテスト
	m_C, _ := linalg.NewMatrix(2, 2, 1, 2, 3, 4)
	_, err = m_A.Multiply(m_C) // A(2x3) * C(2x2) -> 3 != 2 でエラー
	if err == nil {
		t.Error("乗算ルール違反の場合にエラーが返されませんでした")
	}
}