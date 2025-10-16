package zimmat

import "fmt"

// Version は zimmat ライブラリの現在のバージョンです。
const Version = "0.1.0"

// Precision はライブラリ全体で使用される浮動小数点数の許容誤差（ε）です。
// 通常、3D処理では float64（倍精度）が推奨されますが、用途に応じて変更できます。
const Precision = 1e-9

// Init はライブラリが使用される前に、任意の初期設定を行うための関数です。
// 現時点では単に初期化メッセージを表示するだけですが、将来的にCPUのSIMD機能チェックや
// グローバルな設定読み込みなどに利用できます。
func Init() {
	fmt.Println("zimmat library initialized (Version:", Version, ")")
	// 例: 環境変数の読み込み、ハードウェアの互換性チェックなど
}

// CheckVersion は現在のライブラリバージョンを返すシンプルなヘルパー関数です。
func CheckVersion() string {
	return Version
}

/*
他のファイル（例: linalg/vector.go）のコードは、
以下のようにインポートすることで利用されます。

// 例:
// import "zimmat/linalg"
// func New3DVector(x, y, z float64) linalg.Vector {
//     return linalg.NewVector(x, y, z)
// }
*/
