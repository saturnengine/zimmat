package linalg

import (
	"fmt"
	"math"
)

// Vector はn次元のベクトルを表す汎用的な構造体です。
type Vector struct {
	Data []float64 // ベクトルの要素
	Dim  int       // 次元数
}

// NewVector は指定された要素から新しい Vector を作成します。
func NewVector(elements ...float64) Vector {
	return Vector{
		Data: elements,
		Dim:  len(elements),
	}
}

// Add は現在のベクトルに別のベクトルを加算した新しいベクトルを返します。
// 次元が一致しない場合はエラーを返します。
func (v Vector) Add(other Vector) (Vector, error) {
	if v.Dim != other.Dim {
		return Vector{}, fmt.Errorf("次元数が異なります: %d と %d", v.Dim, other.Dim)
	}

	result := make([]float64, v.Dim)
	for i := 0; i < v.Dim; i++ {
		result[i] = v.Data[i] + other.Data[i]
	}

	return NewVector(result...), nil
}

// Dot は2つのベクトルの内積（ドット積）を計算します。
// 次元が一致しない場合はエラーを返します。
func (v Vector) Dot(other Vector) (float64, error) {
	if v.Dim != other.Dim {
		return 0, fmt.Errorf("次元数が異なります: %d と %d", v.Dim, other.Dim)
	}

	var sum float64
	for i := 0; i < v.Dim; i++ {
		sum += v.Data[i] * other.Data[i]
	}
	return sum, nil
}

// LengthSq はベクトルの長さ（ノルム）の二乗を計算します。
func (v Vector) LengthSq() float64 {
	sq, _ := v.Dot(v)
	return sq
}

// Length はベクトルの長さ（ノルム）を計算します。
func (v Vector) Length() float64 {
	return math.Sqrt(v.LengthSq())
}

// Normalize はベクトルの正規化（長さ1の単位ベクトル化）を行った新しいベクトルを返します。
// ゼロベクトル（長さ0）の場合はエラーを返します。
func (v Vector) Normalize() (Vector, error) {
	length := v.Length()
	if length == 0 {
		return Vector{}, fmt.Errorf("ゼロベクトルは正規化できません")
	}

	result := make([]float64, v.Dim)
	for i := 0; i < v.Dim; i++ {
		result[i] = v.Data[i] / length
	}

	return NewVector(result...), nil
}