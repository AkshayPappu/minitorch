#pragma once

#include <minitorch/dtype.hpp>
#include <minitorch/storage.hpp>

#include <vector>
#include <string>
#include <memory>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <sstream>
#include <cassert>

namespace minitorch {

class GradFunction;

class Tensor {
public:
    // ── Phase 1: Default constructor ──
    Tensor();

    // ── Phase 2: Constructors / Creation ──
    explicit Tensor(std::vector<int> shape);
    Tensor(std::vector<int> shape, float value);
    static Tensor from_blob(float* ptr, std::vector<int> shape);
    Tensor clone() const;

    static Tensor zeros(std::vector<int> shape);
    static Tensor ones(std::vector<int> shape);
    static Tensor empty(std::vector<int> shape);
    static Tensor full(std::vector<int> shape, float value);
    static Tensor arange(float start, float end, float step = 1.0f);
    static Tensor eye(int n);
    static Tensor diag(const Tensor& tensor);

    // ── Phase 1: Core Metadata ──
    const std::vector<int>& sizes() const;
    const std::vector<int>& strides() const;
    int dim() const;
    int numel() const;
    DType dtype() const;
    Device device() const;
    int storage_offset() const;
    bool is_contiguous() const;
    bool is_empty() const;

    // ── Phase 1: Storage Access ──
    float* data_ptr() const;
    const Storage& storage() const;
    void set_storage(const Storage& s);
    void set_storage_offset(int offset);

    // ── Phase 3: Indexing / Access ──
    Tensor operator[](int index) const;
    float at(std::initializer_list<int> indices) const;
    void set(std::initializer_list<int> indices, float value);
    float item() const;

    // ── Phase 4: View / Shape Ops ──
    Tensor view(std::vector<int> shape) const;
    Tensor reshape(std::vector<int> shape) const;
    Tensor flatten() const;
    Tensor transpose(int dim0, int dim1) const;
    Tensor permute(std::vector<int> order) const;
    Tensor squeeze(int dim = -1) const;
    Tensor unsqueeze(int dim) const;
    Tensor expand(std::vector<int> shape) const;
    Tensor narrow(int dim, int start, int length) const;
    Tensor slice(int dim, int start, int end, int step = 1) const;
    Tensor select(int dim, int index) const;
    Tensor contiguous() const;

    // ── Phase 5: Copy / Ownership ──
    void copy_(const Tensor& other);
    Tensor detach() const;
    bool is_shared_storage() const;

    // ── Phase 6: Elementwise Ops ──
    Tensor add(const Tensor& other) const;
    Tensor sub(const Tensor& other) const;
    Tensor mul(const Tensor& other) const;
    Tensor div(const Tensor& other) const;
    Tensor pow(const Tensor& other) const;
    Tensor neg() const;
    Tensor exp() const;
    Tensor log() const;
    Tensor sqrt() const;
    Tensor abs() const;
    Tensor relu() const;
    Tensor sigmoid() const;
    Tensor tanh() const;

    Tensor add_scalar(float value) const;
    Tensor sub_scalar(float value) const;
    Tensor mul_scalar(float value) const;
    Tensor div_scalar(float value) const;
    Tensor pow_scalar(float value) const;

    Tensor& add_(const Tensor& other);
    Tensor& sub_(const Tensor& other);
    Tensor& mul_(const Tensor& other);
    Tensor& div_(const Tensor& other);
    Tensor& add_scalar_(float value);
    Tensor& mul_scalar_(float value);

    // ── Phase 7: Reduction Ops ──
    Tensor sum(int dim = -1) const;
    Tensor mean(int dim = -1) const;
    Tensor max(int dim = -1) const;
    Tensor min(int dim = -1) const;
    Tensor argmax(int dim = -1) const;
    Tensor argmin(int dim = -1) const;

    // ── Phase 8: Comparison Ops ──
    Tensor eq(const Tensor& other) const;
    Tensor ne(const Tensor& other) const;
    Tensor lt(const Tensor& other) const;
    Tensor le(const Tensor& other) const;
    Tensor gt(const Tensor& other) const;
    Tensor ge(const Tensor& other) const;

    // ── Phase 8: Linear Algebra ──
    Tensor matmul(const Tensor& other) const;
    Tensor mm(const Tensor& other) const;
    Tensor bmm(const Tensor& other) const;
    Tensor dot(const Tensor& other) const;
    Tensor outer(const Tensor& other) const;

    // ── Phase 8: Concatenation / Splitting ──
    static Tensor cat(const std::vector<Tensor>& tensors, int dim = 0);
    static Tensor stack(const std::vector<Tensor>& tensors, int dim = 0);
    std::vector<Tensor> split(int size, int dim = 0) const;
    std::vector<Tensor> chunk(int chunks, int dim = 0) const;

    // ── Phase 9: Autograd ──
    bool requires_grad() const;
    void set_requires_grad(bool req);
    Tensor grad() const;
    void set_grad(const Tensor& g);
    std::shared_ptr<GradFunction> grad_fn() const;
    void backward();
    void backward(const Tensor& grad_output);
    void zero_grad();

    // ── Phase 10: Utility / Debug ──
    void print(std::ostream& os = std::cout) const;
    std::string to_string() const;
    std::string shape_string() const;
    std::string stride_string() const;

    Tensor to(Device dev) const;
    Tensor astype(DType dt) const;
    Tensor pin_memory() const;

    friend std::ostream& operator<<(std::ostream& os, const Tensor& t);

private:
    Storage storage_;
    std::vector<int> shape_;
    std::vector<int> strides_;
    int offset_ = 0;
    DType dtype_ = DType::Float32;
    Device device_ = Device::CPU;

    // Autograd fields
    bool requires_grad_ = false;
    std::shared_ptr<Tensor> grad_;
    std::shared_ptr<GradFunction> grad_fn_;

    // Helpers
    int flat_index(std::initializer_list<int> indices) const;
    static std::vector<int> compute_strides(const std::vector<int>& shape);
    static int compute_numel(const std::vector<int>& shape);
    Tensor apply_unary(std::function<float(float)> fn) const;
    Tensor apply_binary(const Tensor& other, std::function<float(float, float)> fn) const;
    Tensor& apply_binary_inplace(const Tensor& other, std::function<float(float, float)> fn);
    Tensor reduce(int dim, std::function<float(float, float)> fn, float init) const;
    Tensor reduce_arg(int dim, std::function<bool(float, float)> cmp) const;

    static std::pair<std::vector<int>, std::vector<std::pair<std::vector<int>, std::vector<int>>>>
    broadcast_shapes(const Tensor& a, const Tensor& b);
};

} // namespace minitorch
