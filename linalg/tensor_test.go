package linalg_test

import (
	"testing"
	"github.com/saturnengine/zimmat/linalg"
)

// TestNewTensor はNewTensor関数のテストです。
func TestNewTensor(t *testing.T) {
	// 3次元テンソルの作成 (2x3x4)
	tensor := linalg.NewTensor(2, 3, 4)
	
	if tensor.Rank != 3 {
		t.Errorf("テンソルの階数が期待値と異なります。期待値: 3, 実際: %d", tensor.Rank)
	}
	
	expectedShape := []int{2, 3, 4}
	for i, dim := range tensor.Shape {
		if dim != expectedShape[i] {
			t.Errorf("テンソルの形状[%d]が期待値と異なります。期待値: %d, 実際: %d", i, expectedShape[i], dim)
		}
	}
	
	if tensor.Size() != 24 {
		t.Errorf("テンソルのサイズが期待値と異なります。期待値: 24, 実際: %d", tensor.Size())
	}
}

// TestNewTensorWithData はNewTensorWithData関数のテストです。
func TestNewTensorWithData(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, err := linalg.NewTensorWithData(data, 2, 3)
	
	if err != nil {
		t.Fatalf("NewTensorWithDataでエラーが発生しました: %v", err)
	}
	
	if tensor.Rank != 2 {
		t.Errorf("テンソルの階数が期待値と異なります。期待値: 2, 実際: %d", tensor.Rank)
	}
	
	val, _ := tensor.Get(1, 2)
	if !almostEqual(val, 6.0) {
		t.Errorf("テンソルの要素[1,2]が期待値と異なります。期待値: 6.0, 実際: %f", val)
	}
}

// TestTensorGetSet はGet/Setメソッドのテストです。
func TestTensorGetSet(t *testing.T) {
	tensor := linalg.NewTensor(3, 3)
	
	// Setのテスト
	err := tensor.Set(42.0, 1, 2)
	if err != nil {
		t.Fatalf("Set操作でエラーが発生しました: %v", err)
	}
	
	// Getのテスト
	val, err := tensor.Get(1, 2)
	if err != nil {
		t.Fatalf("Get操作でエラーが発生しました: %v", err)
	}
	
	if !almostEqual(val, 42.0) {
		t.Errorf("取得した値が期待値と異なります。期待値: 42.0, 実際: %f", val)
	}
	
	// 範囲外アクセスのテスト
	_, err = tensor.Get(3, 0)
	if err == nil {
		t.Error("範囲外アクセスでエラーが返されませんでした")
	}
}

// TestTensorAdd はAdd演算のテストです。
func TestTensorAdd(t *testing.T) {
	data1 := []float64{1, 2, 3, 4}
	data2 := []float64{5, 6, 7, 8}
	
	tensor1, _ := linalg.NewTensorWithData(data1, 2, 2)
	tensor2, _ := linalg.NewTensorWithData(data2, 2, 2)
	
	result, err := tensor1.Add(tensor2)
	if err != nil {
		t.Fatalf("Add操作でエラーが発生しました: %v", err)
	}
	
	expected := []float64{6, 8, 10, 12}
	for i, val := range result.Data {
		if !almostEqual(val, expected[i]) {
			t.Errorf("加算結果[%d]が期待値と異なります。期待値: %f, 実際: %f", i, expected[i], val)
		}
	}
}

// TestTensorSubtract はSubtract演算のテストです。
func TestTensorSubtract(t *testing.T) {
	data1 := []float64{5, 6, 7, 8}
	data2 := []float64{1, 2, 3, 4}
	
	tensor1, _ := linalg.NewTensorWithData(data1, 2, 2)
	tensor2, _ := linalg.NewTensorWithData(data2, 2, 2)
	
	result, err := tensor1.Subtract(tensor2)
	if err != nil {
		t.Fatalf("Subtract操作でエラーが発生しました: %v", err)
	}
	
	expected := []float64{4, 4, 4, 4}
	for i, val := range result.Data {
		if !almostEqual(val, expected[i]) {
			t.Errorf("減算結果[%d]が期待値と異なります。期待値: %f, 実際: %f", i, expected[i], val)
		}
	}
}

// TestTensorScale はScale演算のテストです。
func TestTensorScale(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	tensor, _ := linalg.NewTensorWithData(data, 2, 2)
	
	result := tensor.Scale(2.0)
	
	expected := []float64{2, 4, 6, 8}
	for i, val := range result.Data {
		if !almostEqual(val, expected[i]) {
			t.Errorf("スケール結果[%d]が期待値と異なります。期待値: %f, 実際: %f", i, expected[i], val)
		}
	}
}

// TestTensorReshape はReshape操作のテストです。
func TestTensorReshape(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, _ := linalg.NewTensorWithData(data, 2, 3)
	
	reshaped, err := tensor.Reshape(3, 2)
	if err != nil {
		t.Fatalf("Reshape操作でエラーが発生しました: %v", err)
	}
	
	if reshaped.Rank != 2 {
		t.Errorf("リシェイプ後の階数が期待値と異なります。期待値: 2, 実際: %d", reshaped.Rank)
	}
	
	expectedShape := []int{3, 2}
	for i, dim := range reshaped.Shape {
		if dim != expectedShape[i] {
			t.Errorf("リシェイプ後の形状[%d]が期待値と異なります。期待値: %d, 実際: %d", i, expectedShape[i], dim)
		}
	}
	
	// データが保持されているかチェック
	val, _ := reshaped.Get(1, 1)
	if !almostEqual(val, 4.0) {
		t.Errorf("リシェイプ後の要素[1,1]が期待値と異なります。期待値: 4.0, 実際: %f", val)
	}
}

// TestTensorTranspose は転置操作のテストです。
func TestTensorTranspose(t *testing.T) {
	data := []float64{1, 2, 3, 4, 5, 6}
	tensor, _ := linalg.NewTensorWithData(data, 2, 3) // 2x3行列
	
	transposed, err := tensor.Transpose()
	if err != nil {
		t.Fatalf("Transpose操作でエラーが発生しました: %v", err)
	}
	
	if transposed.Rank != 2 {
		t.Errorf("転置後の階数が期待値と異なります。期待値: 2, 実際: %d", transposed.Rank)
	}
	
	expectedShape := []int{3, 2}
	for i, dim := range transposed.Shape {
		if dim != expectedShape[i] {
			t.Errorf("転置後の形状[%d]が期待値と異なります。期待値: %d, 実際: %d", i, expectedShape[i], dim)
		}
	}
	
	// 転置が正しく行われているかチェック
	val, _ := transposed.Get(1, 0)
	originalVal, _ := tensor.Get(0, 1)
	if !almostEqual(val, originalVal) {
		t.Errorf("転置後の要素[1,0]が期待値と異なります。期待値: %f, 実際: %f", originalVal, val)
	}
}

// TestTensorMatrixMultiply は行列乗算のテストです。
func TestTensorMatrixMultiply(t *testing.T) {
	// A (2x3) * B (3x2) = C (2x2)
	dataA := []float64{1, 2, 3, 4, 5, 6}
	dataB := []float64{7, 8, 9, 10, 11, 12}
	
	tensorA, _ := linalg.NewTensorWithData(dataA, 2, 3)
	tensorB, _ := linalg.NewTensorWithData(dataB, 3, 2)
	
	result, err := tensorA.MatrixMultiply(tensorB)
	if err != nil {
		t.Fatalf("MatrixMultiply操作でエラーが発生しました: %v", err)
	}
	
	// 期待される結果: [[58, 64], [139, 154]]
	expected := []float64{58, 64, 139, 154}
	for i, val := range result.Data {
		if !almostEqual(val, expected[i]) {
			t.Errorf("行列乗算結果[%d]が期待値と異なります。期待値: %f, 実際: %f", i, expected[i], val)
		}
	}
}

// TestTensorVectorDot はベクトル内積のテストです。
func TestTensorVectorDot(t *testing.T) {
	data1 := []float64{1, 2, 3}
	data2 := []float64{4, 5, 6}
	
	tensor1, _ := linalg.NewTensorWithData(data1, 3)
	tensor2, _ := linalg.NewTensorWithData(data2, 3)
	
	dot, err := tensor1.VectorDot(tensor2)
	if err != nil {
		t.Fatalf("VectorDot操作でエラーが発生しました: %v", err)
	}
	
	expected := 32.0 // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
	if !almostEqual(dot, expected) {
		t.Errorf("内積の結果が期待値と異なります。期待値: %f, 実際: %f", expected, dot)
	}
}

// TestTensorVectorLength はベクトル長のテストです。
func TestTensorVectorLength(t *testing.T) {
	data := []float64{3, 4} // 長さ5のベクトル
	tensor, _ := linalg.NewTensorWithData(data, 2)
	
	length, err := tensor.VectorLength()
	if err != nil {
		t.Fatalf("VectorLength操作でエラーが発生しました: %v", err)
	}
	
	expected := 5.0
	if !almostEqual(length, expected) {
		t.Errorf("ベクトル長が期待値と異なります。期待値: %f, 実際: %f", expected, length)
	}
}

// TestTensorVectorNormalize はベクトル正規化のテストです。
func TestTensorVectorNormalize(t *testing.T) {
	data := []float64{3, 4} // 長さ5のベクトル
	tensor, _ := linalg.NewTensorWithData(data, 2)
	
	normalized, err := tensor.VectorNormalize()
	if err != nil {
		t.Fatalf("VectorNormalize操作でエラーが発生しました: %v", err)
	}
	
	// 正規化されたベクトルの長さが1であることを確認
	length, _ := normalized.VectorLength()
	if !almostEqual(length, 1.0) {
		t.Errorf("正規化されたベクトルの長さが1ではありません: %f", length)
	}
	
	// 正規化されたベクトルの要素が正しいかチェック
	val0, _ := normalized.Get(0)
	val1, _ := normalized.Get(1)
	if !almostEqual(val0, 0.6) || !almostEqual(val1, 0.8) {
		t.Errorf("正規化されたベクトルの要素が期待値と異なります。期待値: [0.6, 0.8], 実際: [%f, %f]", val0, val1)
	}
}

// TestTensorAsVectorMatrix はAsVector/AsMatrixメソッドのテストです。
func TestTensorAsVectorMatrix(t *testing.T) {
	// 1次元テンソルからベクトルへの変換
	data1d := []float64{1, 2, 3}
	tensor1d, _ := linalg.NewTensorWithData(data1d, 3)
	
	vector, err := tensor1d.AsVector()
	if err != nil {
		t.Fatalf("AsVector変換でエラーが発生しました: %v", err)
	}
	
	if vector.Dim != 3 {
		t.Errorf("変換されたベクトルの次元数が期待値と異なります。期待値: 3, 実際: %d", vector.Dim)
	}
	
	// 2次元テンソルから行列への変換
	data2d := []float64{1, 2, 3, 4}
	tensor2d, _ := linalg.NewTensorWithData(data2d, 2, 2)
	
	matrix, err := tensor2d.AsMatrix()
	if err != nil {
		t.Fatalf("AsMatrix変換でエラーが発生しました: %v", err)
	}
	
	if matrix.Rows != 2 || matrix.Cols != 2 {
		t.Errorf("変換された行列のサイズが期待値と異なります。期待値: 2x2, 実際: %dx%d", matrix.Rows, matrix.Cols)
	}
}

// TestTensorClone はCloneメソッドのテストです。
func TestTensorClone(t *testing.T) {
	data := []float64{1, 2, 3, 4}
	original, _ := linalg.NewTensorWithData(data, 2, 2)
	
	cloned := original.Clone()
	
	// クローンが同じデータを持つかチェック
	for i := range original.Data {
		if !almostEqual(original.Data[i], cloned.Data[i]) {
			t.Errorf("クローンのデータ[%d]が異なります。期待値: %f, 実際: %f", i, original.Data[i], cloned.Data[i])
		}
	}
	
	// 独立性を確認（一方を変更しても他方が影響されない）
	original.Set(99.0, 0, 0)
	clonedVal, _ := cloned.Get(0, 0)
	if almostEqual(clonedVal, 99.0) {
		t.Error("クローンが独立していません。オリジナルの変更がクローンに影響しています")
	}
}