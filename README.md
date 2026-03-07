# minitorch

A lightweight C++20 tensor library inspired by PyTorch. Supports N-dimensional tensors with shared-storage views, broadcasting, linear algebra, autograd, and more -- all from scratch with zero external dependencies beyond the standard library.

## Quick Start

```cpp
#include <minitorch/tensor.hpp>
using namespace minitorch;

int main() {
    // Create tensors
    Tensor a = Tensor::arange(0, 12, 1).view({3, 4});
    Tensor b = Tensor::ones({3, 4});

    // Elementwise math with broadcasting
    Tensor c = a.add(b).mul_scalar(2.0f);

    // Matrix multiply
    Tensor x = Tensor::eye(3);
    Tensor y = Tensor::arange(1, 4, 1);
    Tensor z = x.matmul(y);  // [1, 2, 3]

    // Reductions
    Tensor s = a.sum(1);     // sum along columns
    Tensor m = a.mean();     // global mean

    // Views share storage -- no copies
    Tensor row = a[0];       // first row (view)
    Tensor t = a.transpose(0, 1);

    // Print
    std::cout << c << std::endl;
}
```

## Requirements

- **CMake** 3.16+
- **C++20 compiler** -- GCC 10+, Clang 10+, or MSVC 2019+
- **Windows only:** [MSYS2](https://www.msys2.org/) with MinGW (`pacman -S mingw-w64-ucrt-x86_64-gcc`)

## Build

### Windows (MSYS2 MinGW)

```powershell
.\configure.ps1
cd build
cmake --build .
```

### Linux / macOS

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Run Tests

```bash
# Windows
.\build\minitorch_tests.exe

# Linux / macOS
./build/minitorch_tests

# Or via ctest
cd build && ctest
```

## Features

### Tensor Creation

| Function | Description |
|---|---|
| `Tensor(shape)` | Uninitialized tensor with given shape |
| `Tensor(shape, value)` | Filled with a constant value |
| `Tensor::zeros(shape)` | All zeros |
| `Tensor::ones(shape)` | All ones |
| `Tensor::full(shape, val)` | Filled with `val` |
| `Tensor::empty(shape)` | Uninitialized (allocated but not zeroed) |
| `Tensor::arange(start, end, step)` | Evenly spaced values |
| `Tensor::eye(n)` | n x n identity matrix |
| `Tensor::diag(tensor)` | Diagonal matrix from a 1D tensor |
| `Tensor::from_blob(ptr, shape)` | Wrap existing memory (no copy) |
| `clone()` | Deep copy |

### Metadata

| Function | Returns |
|---|---|
| `sizes()` | Shape as `vector<int>` |
| `strides()` | Strides as `vector<int>` |
| `dim()` | Number of dimensions |
| `numel()` | Total element count |
| `dtype()` | Data type enum |
| `device()` | Device enum (CPU/CUDA) |
| `storage_offset()` | Offset into underlying storage |
| `is_contiguous()` | Whether memory layout is contiguous |
| `is_empty()` | Whether any dimension is zero |

### Indexing and Element Access

```cpp
Tensor t = Tensor::arange(0, 12, 1).view({3, 4});

Tensor row = t[0];              // view of first row
float val  = t.at({1, 2});     // element at (1, 2)
t.set({1, 2}, 99.0f);          // set element
float s    = t.item();          // scalar from 1-element tensor
```

### View and Shape Operations

All view ops share storage with the original tensor -- no data is copied.

```cpp
Tensor a = Tensor::arange(0, 24, 1);

a.view({2, 3, 4});         // reshape (must be contiguous)
a.reshape({4, 6});          // reshape (copies if needed)
a.flatten();                // 1D view
a.view({6, 4}).transpose(0, 1);  // swap dimensions
a.view({6, 4}).permute({1, 0});  // arbitrary axis reorder

Tensor b({1, 3, 1, 4}, 1.0f);
b.squeeze();                // remove all size-1 dims -> (3, 4)
b.squeeze(0);               // remove dim 0 only -> (3, 1, 4)

Tensor c({3, 4}, 1.0f);
c.unsqueeze(0);             // add dim -> (1, 3, 4)
c.expand({5, 3, 4});        // broadcast without copying

a.narrow(0, 2, 5);          // elements [2..7)
a.slice(0, 0, 10, 2);       // every other element
a.view({6, 4}).select(0, 1); // select row 1
a.view({6, 4}).contiguous(); // force contiguous copy if needed
```

### Elementwise Operations

```cpp
Tensor a = Tensor::arange(1, 5, 1);
Tensor b = Tensor::full({4}, 2.0f);

// Tensor-tensor (with broadcasting)
a.add(b);  a.sub(b);  a.mul(b);  a.div(b);  a.pow(b);

// Unary
a.neg();  a.exp();  a.log();  a.sqrt();  a.abs();

// Activations
a.relu();  a.sigmoid();  a.tanh();

// Scalar
a.add_scalar(10.0f);  a.mul_scalar(0.5f);

// In-place (modifies tensor directly)
a.add_(b);  a.mul_(b);  a.add_scalar_(1.0f);
```

### Reduction Operations

```cpp
Tensor t = Tensor::arange(0, 12, 1).view({3, 4});

t.sum();        // global sum -> scalar
t.sum(0);       // sum along dim 0 -> shape (4,)
t.sum(1);       // sum along dim 1 -> shape (3,)
t.mean();       // global mean
t.max(0);       // max along dim 0
t.min(1);       // min along dim 1
t.argmax();     // index of global max
t.argmin(0);    // argmin along dim 0
```

### Comparison Operations

Return tensors of 0.0 / 1.0:

```cpp
a.eq(b);  a.ne(b);  a.lt(b);  a.le(b);  a.gt(b);  a.ge(b);
```

### Linear Algebra

```cpp
Tensor A = Tensor::eye(3);
Tensor B = Tensor::arange(0, 9, 1).view({3, 3});

A.mm(B);                    // 2D matrix multiply
A.matmul(B);                // general matmul (1D/2D/3D dispatch)

Tensor x = Tensor::arange(1, 4, 1);
Tensor y = Tensor::arange(1, 4, 1);
x.dot(y);                   // inner product
x.outer(y);                 // outer product

// Batched matmul
Tensor P = Tensor::ones({2, 3, 4});
Tensor Q = Tensor::ones({2, 4, 5});
P.bmm(Q);                   // -> (2, 3, 5)
```

### Concatenation and Splitting

```cpp
Tensor a = Tensor::ones({2, 3});
Tensor b = Tensor::zeros({2, 3});

Tensor::cat({a, b}, 0);     // concat along dim 0 -> (4, 3)
Tensor::stack({a, b}, 0);   // stack -> (2, 2, 3)

auto parts = a.split(1, 1); // split into size-1 chunks along dim 1
auto chunks = a.chunk(2, 0); // split into 2 chunks along dim 0
```

### Copy and Ownership

```cpp
Tensor a({3, 3}, 1.0f);
Tensor b = a[0];             // b is a view -- shared storage
a.is_shared_storage();       // true

Tensor c = a.clone();        // deep copy -- independent storage
c.is_shared_storage();       // false

Tensor d({3, 3}, 5.0f);
a.copy_(d);                  // copy d's data into a (in-place)

Tensor e = a.detach();       // shares storage, detached from autograd
```

### Autograd

```cpp
Tensor w({2, 2}, 1.0f);
w.set_requires_grad(true);

// Accumulate gradients
w.backward(Tensor({2, 2}, 0.5f));
std::cout << w.grad() << std::endl;  // all 0.5

// Gradients accumulate across calls
w.backward(Tensor({2, 2}, 1.0f));
std::cout << w.grad() << std::endl;  // all 1.5

w.zero_grad();               // reset gradients
Tensor d = w.detach();       // sever autograd connection
```

### Utility and Debug

```cpp
Tensor t = Tensor::arange(0, 6, 1).view({2, 3});

std::cout << t << std::endl;         // pretty print
std::string s = t.to_string();       // string representation
t.shape_string();                    // "(2, 3)"
t.stride_string();                   // "(3, 1)"

t.to(Device::CUDA);                  // device transfer (stub)
t.astype(DType::Float64);            // dtype cast (stub)
t.pin_memory();                      // pin memory (stub/no-op)
```

## Project Structure

```
minitorch/
├── include/minitorch/
│   ├── tensor.hpp       # Tensor class (all public API)
│   ├── storage.hpp      # Reference-counted storage backend
│   └── dtype.hpp        # DType and Device enums
├── src/
│   └── tensor.cpp       # Full implementation
├── tests/
│   └── tensor_test.cpp  # 70 GoogleTest tests (all passing)
├── CMakeLists.txt        # Build configuration
└── configure.ps1         # Windows MSYS2 setup script
```

## Architecture

- **Storage** -- `shared_ptr<float[]>` enables reference-counted memory. Views (transpose, slice, select, etc.) share the same storage with different shape/strides/offset -- zero-copy.
- **Strides** -- Stored explicitly on every tensor. Non-contiguous layouts (from transpose, permute, etc.) are first-class citizens. Operations that require contiguous data call `contiguous()` internally.
- **Broadcasting** -- Binary operations automatically broadcast compatible shapes by inserting stride-0 dimensions, matching NumPy/PyTorch semantics.
- **Autograd** -- Gradient storage and accumulation via `backward()`. The `GradFunction` base class is defined for future computation graph support.

## License

MIT
