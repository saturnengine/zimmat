package linalg_test

import (
	"math"
	"testing"

	//"zimmat/linalg" // モジュール名に合わせてインポートパスを変更してください
	"github.com/saturnengine/zimmat.git/linalg"
)

// floatの比較で許容される誤差 (epsilon)
const floatTolerance = 1e-9

// almostEqual は浮動小数点数を許容誤差内で比較します。
func almostEqual(a, b float64) bool {
	return math.Abs(a-b) < floatTolerance
}

// testVectorsEqual は2つのベクトルが等しいか（要素と次元が許容誤差内で一致するか）をチェックします。
func testVectorsEqual(v1, v2 linalg.Vector) bool {
	if v1.Dim != v2.Dim {
		return false
	}
	for i := 0; i < v1.Dim; i++ {
		if !almostEqual(v1.Data[i], v2.Data[i]) {
			return false
		}
	}
	return true
}

// TestNewVector はNewVector関数のテストです。
func TestNewVector(t *testing.T) {
	v := linalg.NewVector(1.0, 2.0, 3.0)
	if v.Dim != 3 {
		t.Errorf("次元数が期待値と異なります。期待値: 3, 実際: %d", v.Dim)
	}
	expected := []float64{1.0, 2.0, 3.0}
	for i := range expected {
		if !almostEqual(v.Data[i], expected[i]) {
			t.Errorf("要素が期待値と異なります。期待値: %v, 実際: %v", expected, v.Data)
			break
		}
	}
}

// TestVectorAdd はベクトルの加算メソッドのテストです。
func TestVectorAdd(t *testing.T) {
	v1 := linalg.NewVector(1, 2, 3)
	v2 := linalg.NewVector(4, 5, 6)
	expected := linalg.NewVector(5, 7, 9)

	result, err := v1.Add(v2)
	if err != nil {
		t.Fatalf("Addでエラーが発生しました: %v", err)
	}

	if !testVectorsEqual(result, expected) {
		t.Errorf("Addの結果が期待値と異なります。期待値: %v, 実際: %v", expected.Data, result.Data)
	}

	// 次元不一致のテスト
	v3 := linalg.NewVector(1, 2)
	_, err = v1.Add(v3)
	if err == nil {
		t.Error("次元不一致の場合にエラーが返されませんでした")
	}
}

// TestVectorDot は内積（ドット積）メソッドのテストです。
func TestVectorDot(t *testing.T) {
	v1 := linalg.NewVector(1, 0, 0)
	v2 := linalg.NewVector(0, 1, 0)
	v3 := linalg.NewVector(2, 3, 4)

	// 垂直ベクトルの内積 (0)
	dot1, _ := v1.Dot(v2)
	if !almostEqual(dot1, 0.0) {
		t.Errorf("内積の計算が誤っています。期待値: 0.0, 実際: %f", dot1)
	}

	// 一般的な内積 (2*2 + 3*3 + 4*4 = 4 + 9 + 16 = 29)
	dot2, _ := v3.Dot(v3)
	if !almostEqual(dot2, 29.0) {
		t.Errorf("内積の計算が誤っています。期待値: 29.0, 実際: %f", dot2)
	}
}

// TestVectorLengthAndNormalize は長さと正規化メソッドのテストです。
func TestVectorLengthAndNormalize(t *testing.T) {
	v := linalg.NewVector(3, 4) // 長さ5のベクトル

	// LengthSqのテスト
	if !almostEqual(v.LengthSq(), 25.0) {
		t.Errorf("LengthSqの結果が誤っています。期待値: 25.0, 実際: %f", v.LengthSq())
	}

	// Lengthのテスト
	if !almostEqual(v.Length(), 5.0) {
		t.Errorf("Lengthの結果が誤っています。期待値: 5.0, 実際: %f", v.Length())
	}

	// Normalizeのテスト
	normalized, err := v.Normalize()
	if err != nil {
		t.Fatalf("Normalizeでエラーが発生しました: %v", err)
	}
	expectedNormalized := linalg.NewVector(0.6, 0.8) // 3/5=0.6, 4/5=0.8

	if !testVectorsEqual(normalized, expectedNormalized) {
		t.Errorf("正規化の結果が期待値と異なります。期待値: %v, 実際: %v", expectedNormalized.Data, normalized.Data)
	}

	// 正規化されたベクトルの長さが1であることを確認
	if !almostEqual(normalized.Length(), 1.0) {
		t.Errorf("正規化されたベクトルの長さが1ではありません: %f", normalized.Length())
	}

	// ゼロベクトルの正規化テスト
	zeroV := linalg.NewVector(0, 0)
	_, err = zeroV.Normalize()
	if err == nil {
		t.Error("ゼロベクトルの正規化でエラーが返されませんでした")
	}
}