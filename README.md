# zimmat: Zero-Inertia Matrix Mathematics

## 🚀 Overview

**`zimmat`** is a high-performance, general-purpose **Linear Algebra** library written in Go. It is designed specifically for domains demanding computational efficiency, such as **3D graphics**, **physics simulations**, and **game development**.

The library name is derived from **Z**ero-**I**nertia **M**atrix **Mat**hematics, reflecting its goal of providing fast and efficient matrix and vector operations with minimal overhead.

-----

## ✨ Features

  * **N-Dimensional Support**: Provides generic data structures (`Vector`, `Matrix`) capable of handling arbitrary dimensions.
  * **3D/4D Optimization**: For maximum performance in 3D graphics, dedicated **fixed-size structs** (`Vec3`, `Vec4`, `Mat4`) will be provided in separate packages in the future.
  * **Go Native**: Implemented purely in Go, avoiding reliance on Cgo or external dependencies, allowing it to leverage Go's concurrency features and memory management.
  * **Robust Error Handling**: Utilizes Go's idiomatic `error` returns for invalid operations, such as dimension mismatches, enhancing code safety.

-----

## ⚙️ Installation

You can add `zimmat` to your project using the standard `go get` command:

```bash
go get github.com/saturnengine/zimmat
```

*Note: Replace with your actual module path if different.*

-----

## 📦 Usage Examples

### 1\. Vector Operations

Use the `linalg` package to create N-dimensional vectors and perform core operations.

```go
package main

import (
	"fmt"
	"github.com/saturnengine/zimmat/linalg"
)

func main() {
	// Create a 3D vector v = (3, 4, 0)
	v := linalg.NewVector(3.0, 4.0, 0.0) 
    
	// Calculate the length (Norm)
	length := v.Length() // 5.0
	fmt.Printf("Length of V: %.1f\n", length)

	// Normalize (create a unit vector)
	unitV, err := v.Normalize()
	if err != nil {
		panic(err)
	}
	// unitV = (0.6, 0.8, 0.0)
	fmt.Printf("Normalized V: %v\n", unitV.Data)

	// Calculate the Dot Product
	v2 := linalg.NewVector(1.0, 0.0, 0.0)
	dot, _ := v.Dot(v2) 
	fmt.Printf("Dot Product: %.1f\n", dot) // 3.0
}
```

### 2\. Matrix Operations

The [`Matrix`](linalg/matrix.go:8-14) struct is fully implemented with comprehensive linear algebra operations.

```go
package main

import (
	"fmt"
	"github.com/saturnengine/zimmat/linalg"
)

func main() {
	// Create a 3x3 matrix
	data := [][]float64{
		{1.0, 2.0, 3.0},
		{4.0, 5.0, 6.0},
		{7.0, 8.0, 9.0},
	}
	matrix, err := linalg.NewMatrix(data)
	if err != nil {
		panic(err)
	}

	// Matrix operations
	transposed := matrix.Transpose()
	fmt.Printf("Transposed matrix: %dx%d\n", transposed.Rows, transposed.Cols)

	// Matrix multiplication
	identity := [][]float64{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}
	identityMatrix, _ := linalg.NewMatrix(identity)
	
	result, err := matrix.Multiply(identityMatrix)
	if err != nil {
		panic(err)
	}
	fmt.Printf("Matrix multiplication result: %dx%d\n", result.Rows, result.Cols)

	// Calculate determinant (for square matrices)
	det, err := matrix.Determinant()
	if err != nil {
		panic(err)
	}
	fmt.Printf("Determinant: %.1f\n", det)
}
```

-----

## 📚 Development and Testing

If you are developing or testing locally, ensure your `go.mod` module name matches the import path used in your test files.

1.  **Initialize the Go Module** (Run in the project root):

    ```bash
    go mod init github.com/saturnengine/zimmat
    ```

2.  **Run Tests**:

    ```bash
    go test ./...
    ```

    *or*

    ```bash
    go test ./linalg
    ```

-----

## 🤝 Contributing

We welcome bug reports, feature suggestions, and pull requests\! Help us grow `zimmat` into a robust and efficient Go linear algebra library.

-----

## 📜 License

This project is licensed under the **MIT License**.