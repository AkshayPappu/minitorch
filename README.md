# minitorch

A C++20 tensor core library inspired by PyTorch. Built with modern C++ and tested with GoogleTest.

## Requirements

- CMake 3.16+
- C++20 compiler (GCC 10+, Clang 10+, or MSVC 2019+)
- [MSYS2](https://www.msys2.org/) with MinGW (Windows) — for `g++` and `gcc`

## Build

### Windows (MSYS2 MinGW)

```powershell
.\configure.ps1
cd build
cmake --build .
```

### Linux / macOS / Other

```bash
mkdir build && cd build
cmake ..
cmake --build .
```

## Run Tests

```powershell
.\build\minitorch_tests.exe
```

Or with ctest (if available):

```bash
cd build && ctest
```

## Project Structure

```
minitorch/
├── include/minitorch/   # Public headers
├── src/                 # Implementation
├── tests/               # GoogleTest tests
└── build/               # Build output
```
