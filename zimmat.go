// Package zimmat provides high-performance linear algebra operations for Go.
//
// zimmat (Zero-Inertia Matrix Mathematics) is designed specifically for domains
// demanding computational efficiency, such as 3D graphics, physics simulations,
// and game development. The library provides N-dimensional vector and matrix
// operations with robust error handling and Go-native implementation.
//
// Basic usage:
//
//	import "github.com/saturnengine/zimmat"
//	import "github.com/saturnengine/zimmat/linalg"
//
//	func main() {
//		// Initialize the library (optional)
//		zimmat.Init()
//
//		// Create and work with vectors
//		v := linalg.NewVector(3.0, 4.0, 0.0)
//		length := v.Length() // 5.0
//
//		// Create and work with matrices
//		data := [][]float64{{1, 2}, {3, 4}}
//		matrix, err := linalg.NewMatrix(data)
//		if err != nil {
//			panic(err)
//		}
//	}
//
// The main computational functionality is provided by the linalg subpackage,
// which includes Vector, Matrix, and Tensor types with comprehensive
// mathematical operations.
package zimmat

import "fmt"

// Version represents the current version of the zimmat library.
// This constant is used for version checking and compatibility verification.
const Version = "0.1.0"

// Precision defines the floating-point tolerance (epsilon) used throughout
// the library for numerical comparisons. For 3D processing, float64 (double
// precision) is recommended, but this value can be adjusted based on
// specific use cases.
//
// The default value of 1e-9 provides a good balance between precision
// and numerical stability for most applications.
const Precision = 1e-9

// Init performs optional initialization of the zimmat library.
// Currently, it displays an initialization message, but future versions
// may include CPU SIMD feature detection, global configuration loading,
// or other setup operations.
//
// While calling Init is optional, it's recommended for applications that
// want to ensure proper library initialization and to see version information.
//
// Example:
//
//	import "github.com/saturnengine/zimmat"
//
//	func main() {
//		zimmat.Init() // Optional initialization
//		// ... use library functions
//	}
func Init() {
	fmt.Println("zimmat library initialized (Version:", Version, ")")
	// Future: environment variable loading, hardware compatibility checks, etc.
}

// CheckVersion returns the current version string of the zimmat library.
// This function is useful for runtime version verification and debugging.
//
// Returns:
//   - A string representing the current library version (e.g., "0.1.0")
//
// Example:
//
//	version := zimmat.CheckVersion()
//	fmt.Printf("Using zimmat version: %s\n", version)
func CheckVersion() (version string) {
	version = Version
	return
}

/*
Code from other files (e.g., linalg/vector.go) is used by importing as follows:

// Example:
// import "zimmat/linalg"
// func New3DVector(x, y, z float64) linalg.Vector {
//     return linalg.NewVector(x, y, z)
// }
*/
