# minitorch

A lightweight C++20 tensor library inspired by PyTorch -- with a full autograd engine, neural network modules, optimizers, and a working GPT language model. Built entirely from scratch with zero external dependencies beyond the standard library.

```
                    ┌─────────────────────────────────────────────┐
                    │              minitorch stack                 │
                    ├─────────────────────────────────────────────┤
                    │   examples/gpt   GPT model, tokenizer,      │
                    │                  data pipeline, training     │
                    ├─────────────────────────────────────────────┤
                    │   nn + optim     Linear, Embedding, LayerNorm│
                    │                  Dropout, Adam, SGD          │
                    ├─────────────────────────────────────────────┤
                    │   autograd       Computation graph, backward │
                    │                  pass, gradient accumulation │
                    ├─────────────────────────────────────────────┤
                    │   tensor         N-dim tensors, views,       │
                    │                  broadcasting, linalg        │
                    ├─────────────────────────────────────────────┤
                    │   storage        Ref-counted shared memory   │
                    └─────────────────────────────────────────────┘
```

---

## Table of Contents

- [Quick Start](#quick-start)
- [GPT Language Model](#gpt-language-model)
  - [Architecture](#gpt-architecture)
  - [Training](#training)
  - [Configuration](#configuration)
- [Tensor Library](#tensor-library)
- [Neural Network Modules](#neural-network-modules)
- [Autograd Engine](#autograd-engine)
- [Optimizers](#optimizers)
- [Build and Setup](#build-and-setup)
- [Project Structure](#project-structure)
- [License](#license)

---

## Quick Start

```cpp
#include <minitorch/tensor.hpp>
using namespace minitorch;

int main() {
    Tensor a = Tensor::arange(0, 12, 1).view({3, 4});
    Tensor b = Tensor::ones({3, 4});

    Tensor c = a.add(b).mul_scalar(2.0f);   // elementwise with broadcasting
    Tensor z = Tensor::eye(3).matmul(Tensor::arange(1, 4, 1));  // matrix multiply
    Tensor s = a.sum(1);                     // reduction along columns

    std::cout << c << std::endl;
}
```

---

## GPT Language Model

minitorch includes a complete, trainable GPT (Generative Pre-trained Transformer) built entirely on top of the library's tensor operations, autograd, and nn modules. It learns to generate text character-by-character from any input corpus.

### GPT Architecture

```
  Input Token IDs          Position IDs
   (B, T)                   (T,)
     │                        │
     ▼                        ▼
┌──────────┐            ┌──────────┐
│ Token     │            │ Position  │
│ Embedding │            │ Embedding │
│ (V, C)    │            │ (T, C)    │
└────┬──────┘            └────┬──────┘
     │                        │
     └───────── + ────────────┘
               │
               ▼
         ┌──────────┐
         │  Dropout  │
         └────┬──────┘
              │
              ▼
    ┌─────────────────────────────────────┐
    │        Transformer Block  x N       │
    │                                     │
    │   ┌─────────┐                       │
    │   │LayerNorm│──► Causal Self-Attn ──┤
    │   └─────────┘         │             │
    │        └──── + ◄──────┘  (residual) │
    │              │                       │
    │   ┌─────────┐                       │
    │   │LayerNorm│──► MLP ───────────────┤
    │   └─────────┘     │                 │
    │        └──── + ◄──┘      (residual) │
    └─────────────────────────────────────┘
              │
              ▼
         ┌──────────┐
         │ LayerNorm │
         └────┬──────┘
              │
              ▼
         ┌──────────┐
         │  Linear   │  (language model head)
         │ (C, V)    │
         └────┬──────┘
              │
              ▼
        Logits (B, T, V)
```

**Key dimensions:** B = batch size, T = sequence length, C = embedding dim, V = vocab size, N = number of layers.

#### Causal Self-Attention

```
  Input (B, T, C)
        │
        ▼
   ┌──────────┐
   │ Linear    │  (C -> 3C, produces Q, K, V)
   └────┬──────┘
        │
   split into Q, K, V each (B, T, C)
        │
   reshape to (B, n_heads, T, head_dim)
        │
        ▼
   Attention = softmax( Q K^T / sqrt(head_dim) + causal_mask )
        │
        ▼
   Output = Attention * V
        │
   reshape back to (B, T, C)
        │
        ▼
   ┌──────────┐
   │ Linear    │  (C -> C, output projection)
   └────┬──────┘
        │
        ▼
   ┌──────────┐
   │ Dropout   │
   └────┬──────┘
        ▼
   Output (B, T, C)
```

The causal mask ensures each position can only attend to earlier positions, preventing information leakage from future tokens during training.

#### MLP (Feed-Forward Network)

```
  Input (B, T, C)  ──►  Linear(C, 4C)  ──►  GELU  ──►  Linear(4C, C)  ──►  Dropout  ──►  Output
```

The MLP expands the hidden dimension by 4x, applies the GELU activation, then projects back down. This is where most of the model's parameters live.

### Training

Train the GPT on any text file:

```bash
# Build the project first (see Build and Setup below)
cd build

# Train on the included sample text
./examples/gpt/train_gpt ../examples/gpt/data/input.txt

# Or provide your own text file
./examples/gpt/train_gpt path/to/your/text.txt
```

The training loop:
1. **Tokenizes** the input text into characters (character-level tokenizer)
2. **Creates batches** of (input, target) pairs where target is input shifted by one character
3. **Forward pass** through the full transformer
4. **Cross-entropy loss** between predicted and actual next characters
5. **Backward pass** through the autograd computation graph
6. **Adam optimizer** updates all parameters

Sample output after 200 training steps:

```
Loading data from: data/input.txt
Vocab size: 50
Dataset size: 2145 samples
Model: 2 layers, 4 heads, 64 embed dim, 32 block size
Total parameters: 108544

--- Training ---

Step    1 | Loss: 4.1475 | Time: 0.3s
Step   50 | Loss: 3.2054 | Time: 14.0s
Step  100 | Loss: 2.9345 | Time: 29.2s
Step  150 | Loss: 2.8010 | Time: 46.7s
Step  200 | Loss: 2.6071 | Time: 62.2s

=== Final Generation ===
ouren:nu:
Wey
W houe.Ciely vecors lls yaesouelly gafle tr as, e t thecit...
```

### Configuration

All model hyperparameters are set via `GPTConfig`:

| Parameter | Default | Description |
|---|---|---|
| `vocab_size` | 65 | Number of unique tokens (set automatically from data) |
| `n_embed` | 256 | Embedding / hidden dimension (C) |
| `n_heads` | 8 | Number of attention heads |
| `n_layers` | 6 | Number of transformer blocks |
| `block_size` | 128 | Maximum sequence length (T) |
| `dropout` | 0.1 | Dropout probability (0 = no dropout) |

The included `train_gpt.cpp` uses a smaller config for fast iteration:

| Parameter | Value |
|---|---|
| `n_embed` | 64 |
| `n_heads` | 4 |
| `n_layers` | 2 |
| `block_size` | 32 |
| `batch_size` | 4 |
| `learning_rate` | 3e-4 |
| `max_steps` | 200 |

Scale these up for better results on larger datasets.

---

## Tensor Library

### Creation

| Function | Description |
|---|---|
| `Tensor(shape)` | Uninitialized tensor |
| `Tensor(shape, value)` | Filled with a constant |
| `Tensor::zeros(shape)` | All zeros |
| `Tensor::ones(shape)` | All ones |
| `Tensor::full(shape, val)` | Filled with `val` |
| `Tensor::randn(shape)` | Random normal distribution |
| `Tensor::arange(start, end, step)` | Evenly spaced values |
| `Tensor::eye(n)` | Identity matrix |
| `Tensor::tril(n)` / `triu(n)` | Lower / upper triangular |
| `clone()` | Deep copy |

### Indexing

```cpp
Tensor t = Tensor::arange(0, 12, 1).view({3, 4});
Tensor row = t[0];              // view of first row
float val  = t.at({1, 2});     // element at (1, 2)
t.set({1, 2}, 99.0f);          // set element
float s    = t.item();          // scalar from 1-element tensor
```

### Shape Operations

All view ops share storage -- zero copy.

```cpp
a.view({2, 3, 4});             // reshape (contiguous only)
a.reshape({4, 6});              // reshape (copies if needed)
a.transpose(0, 1);             // swap dimensions
a.permute({2, 0, 1});          // arbitrary axis reorder
a.squeeze();                    // remove size-1 dims
a.unsqueeze(0);                 // add dimension
a.expand({5, 3, 4});           // broadcast without copying
a.narrow(0, 2, 5);             // sub-range along dim
a.slice(0, 0, 10, 2);          // strided slice
a.select(0, 1);                // select index along dim
a.contiguous();                 // force contiguous layout
```

### Math

```cpp
// Tensor-tensor (with broadcasting)
a.add(b);  a.sub(b);  a.mul(b);  a.div(b);  a.pow(b);

// Unary
a.neg();  a.exp();  a.log();  a.sqrt();  a.abs();

// Activations
a.relu();  a.sigmoid();  a.tanh();  a.gelu();

// Scalar
a.add_scalar(10.0f);  a.mul_scalar(0.5f);

// In-place
a.add_(b);  a.mul_(b);  a.add_scalar_(1.0f);

// Reductions
a.sum();  a.sum(0);  a.mean();  a.max(0);  a.argmax();

// Comparisons (return 0.0 / 1.0)
a.eq(b);  a.ne(b);  a.lt(b);  a.le(b);  a.gt(b);  a.ge(b);

// Specialized
a.softmax(dim);  a.log_softmax(dim);  a.cross_entropy_loss(targets);
a.layer_norm(weight, bias);  a.embedding_lookup(indices);
a.masked_fill(mask, value);  a.dropout(p, training);  a.variance(dim);
```

### Linear Algebra

```cpp
A.mm(B);                       // 2D matrix multiply
A.matmul(B);                   // general matmul (1D/2D/3D)
x.dot(y);                      // inner product
x.outer(y);                    // outer product
P.bmm(Q);                      // batched matmul

Tensor::cat({a, b}, 0);        // concatenate
Tensor::stack({a, b}, 0);      // stack
a.split(size, dim);             // split into chunks
a.chunk(n, dim);                // split into n pieces
```

---

## Neural Network Modules

PyTorch-style module system with automatic parameter tracking.

```cpp
#include <minitorch/nn.hpp>
using namespace minitorch::nn;
```

| Module | Constructor | Description |
|---|---|---|
| `Linear` | `(in, out, bias=true)` | Fully connected layer \( y = xW^T + b \) |
| `Embedding` | `(num_embeddings, dim)` | Lookup table for token embeddings |
| `LayerNorm` | `(dim, eps=1e-5)` | Layer normalization |
| `Dropout` | `(p=0.1)` | Randomly zeros elements during training |
| `Sequential` | `()` | Chain modules sequentially |

All modules inherit from `Module` and provide:

```cpp
module.forward(input);          // forward pass
module.parameters();            // collect all learnable Tensor* pointers
module.zero_grad();             // reset all gradients
module.train();                 // set training mode
module.eval();                  // set evaluation mode
module.register_parameter(name, tensor);
module.register_module(name, child);
```

---

## Autograd Engine

Define-by-run computation graph with automatic differentiation.

```cpp
Tensor w({2, 2}, 1.0f);
w.set_requires_grad(true);

Tensor y = w.mul_scalar(3.0f).sum();  // builds graph: MulScalar -> Sum
y.backward();                          // propagates gradients

std::cout << w.grad();                 // all 3.0
w.zero_grad();                         // reset for next iteration
```

Every differentiable operation records a `GradFunction` node. Calling `backward()` performs a topological sort of the graph and propagates gradients from output to leaves. Supported backward ops include: `add`, `sub`, `mul`, `div`, `neg`, `exp`, `log`, `sqrt`, `relu`, `sigmoid`, `tanh`, `gelu`, `pow`, `mm`, `bmm`, `sum`, `mean`, `reshape`, `transpose`, `permute`, `expand`, `squeeze`, `unsqueeze`, `select`, `slice`, `cat`, `softmax`, `log_softmax`, `cross_entropy`, `embedding`, `layer_norm`, `masked_fill`.

---

## Optimizers

```cpp
#include <minitorch/optim.hpp>
using namespace minitorch::optim;

auto params = model.parameters();

// SGD with optional momentum
SGD sgd(params, /*lr=*/0.01f, /*momentum=*/0.9f);

// Adam (default betas and epsilon)
Adam adam(params, /*lr=*/3e-4f);

// Training loop
optimizer.zero_grad();
loss.backward();
optimizer.step();
```

---

## Build and Setup

### Requirements

- **CMake** 3.16+
- **C++20 compiler** -- GCC 10+, Clang 10+, or MSVC 2019+
- **Windows only:** [MSYS2](https://www.msys2.org/) with MinGW (`pacman -S mingw-w64-ucrt-x86_64-gcc`)

### Build

```bash
# Windows (MSYS2 MinGW)
.\configure.ps1
cd build
cmake --build .

# Linux / macOS
mkdir build && cd build
cmake ..
cmake --build .
```

### Run Tests

```bash
# Core tensor library tests (70 tests)
./build/minitorch_tests

# GPT and nn module tests (18 tests)
./build/examples/gpt/gpt_tests

# Or all via ctest
cd build && ctest
```

### Train GPT

```bash
cd build
./examples/gpt/train_gpt ../examples/gpt/data/input.txt
```

To train on your own data, provide any plain text file as the argument. The character-level tokenizer automatically builds a vocabulary from the input.

---

## Project Structure

```
minitorch/
├── include/minitorch/
│   ├── tensor.hpp          Tensor class and full public API
│   ├── autograd.hpp        GradFunction base and backward nodes
│   ├── nn.hpp              Module, Linear, Embedding, LayerNorm, Dropout
│   ├── optim.hpp           SGD and Adam optimizers
│   ├── storage.hpp         Reference-counted storage backend
│   └── dtype.hpp           DType and Device enums
├── src/
│   ├── tensor.cpp          Tensor ops + autograd implementations
│   ├── nn.cpp              Module implementations
│   └── optim.cpp           Optimizer implementations
├── tests/
│   └── tensor_test.cpp     70 core tests
├── examples/
│   └── gpt/
│       ├── include/
│       │   ├── gpt.hpp         GPT model (Attention, MLP, TransformerBlock)
│       │   ├── tokenizer.hpp   Character-level tokenizer
│       │   └── dataloader.hpp  TextDataset and DataLoader
│       ├── src/
│       │   ├── gpt.cpp         Model implementation
│       │   ├── tokenizer.cpp   Tokenizer implementation
│       │   └── train_gpt.cpp   Training entry point
│       ├── tests/
│       │   └── gpt_test.cpp    18 GPT / nn tests
│       └── data/
│           └── input.txt       Sample training text
├── CMakeLists.txt
└── configure.ps1               Windows MSYS2 setup script
```

---

## Architecture

- **Storage** -- `shared_ptr<float[]>` enables reference-counted memory. Views (transpose, slice, expand, etc.) share storage with different shape/strides/offset -- zero copy.
- **Strides** -- Stored explicitly on every tensor. Non-contiguous layouts from transpose, permute, etc. are first-class citizens.
- **Broadcasting** -- Binary operations automatically broadcast compatible shapes by inserting stride-0 dimensions, matching NumPy/PyTorch semantics.
- **Autograd** -- Define-by-run computation graph. Each differentiable op records a `GradFunction` node with saved tensors and edges. `backward()` performs iterative topological sort and gradient propagation. `AccumulateGrad` nodes collect gradients for leaf parameters via shared `GradHolder` objects.
- **NN Modules** -- PyTorch-style `Module` base class with recursive parameter collection, child module registration, and train/eval mode switching.
- **Optimizers** -- First-order optimizers (SGD with momentum, Adam) that operate directly on `Tensor*` parameter pointers.

## License

MIT
