#include <minitorch/tensor.hpp>

namespace minitorch {

// ════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════

std::vector<int> Tensor::compute_strides(const std::vector<int>& shape) {
    std::vector<int> strides(shape.size());
    if (shape.empty()) return strides;
    strides.back() = 1;
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * shape[i + 1];
    return strides;
}

int Tensor::compute_numel(const std::vector<int>& shape) {
    if (shape.empty()) return 1;
    int n = 1;
    for (int s : shape) n *= s;
    return n;
}

int Tensor::flat_index(std::initializer_list<int> indices) const {
    if (static_cast<int>(indices.size()) != dim())
        throw std::runtime_error("Index dimension mismatch");
    int idx = offset_;
    auto it = indices.begin();
    for (int i = 0; i < dim(); ++i, ++it) {
        if (*it < 0 || *it >= shape_[i])
            throw std::out_of_range("Index out of bounds");
        idx += (*it) * strides_[i];
    }
    return idx;
}

// ════════════════════════════════════════════════════════════════
// Phase 1: Default Constructor + Core Metadata
// ════════════════════════════════════════════════════════════════

Tensor::Tensor() : storage_(), shape_(), strides_(), offset_(0) {}

const std::vector<int>& Tensor::sizes() const { return shape_; }
const std::vector<int>& Tensor::strides() const { return strides_; }
int Tensor::dim() const { return static_cast<int>(shape_.size()); }

int Tensor::numel() const { return compute_numel(shape_); }

DType Tensor::dtype() const { return dtype_; }
Device Tensor::device() const { return device_; }
int Tensor::storage_offset() const { return offset_; }

bool Tensor::is_contiguous() const {
    if (shape_.empty()) return true;
    int expected = 1;
    for (int i = dim() - 1; i >= 0; --i) {
        if (shape_[i] != 1 && strides_[i] != expected)
            return false;
        expected *= shape_[i];
    }
    return true;
}

bool Tensor::is_empty() const {
    for (int s : shape_)
        if (s == 0) return true;
    return false;
}

// Phase 1: Storage Access
float* Tensor::data_ptr() const { return storage_.data_ptr() + offset_; }
const Storage& Tensor::storage() const { return storage_; }
void Tensor::set_storage(const Storage& s) { storage_ = s; }
void Tensor::set_storage_offset(int offset) { offset_ = offset; }

// ════════════════════════════════════════════════════════════════
// Phase 2: Constructors / Creation
// ════════════════════════════════════════════════════════════════

Tensor::Tensor(std::vector<int> shape)
    : storage_(compute_numel(shape)), shape_(std::move(shape)),
      strides_(compute_strides(shape_)), offset_(0) {}

Tensor::Tensor(std::vector<int> shape, float value)
    : storage_(compute_numel(shape), value), shape_(std::move(shape)),
      strides_(compute_strides(shape_)), offset_(0) {}

Tensor Tensor::from_blob(float* ptr, std::vector<int> shape) {
    Tensor t;
    int n = compute_numel(shape);
    t.storage_ = Storage::from_blob(ptr, n);
    t.shape_ = std::move(shape);
    t.strides_ = compute_strides(t.shape_);
    t.offset_ = 0;
    return t;
}

Tensor Tensor::clone() const {
    Tensor t;
    t.shape_ = shape_;
    t.strides_ = compute_strides(shape_);
    t.offset_ = 0;
    t.dtype_ = dtype_;
    t.device_ = device_;
    t.storage_ = Storage(numel());
    float* dst = t.storage_.data_ptr();
    if (is_contiguous()) {
        std::memcpy(dst, data_ptr(), numel() * sizeof(float));
    } else {
        int n = numel();
        std::vector<int> idx(dim(), 0);
        for (int i = 0; i < n; ++i) {
            int src_off = offset_;
            for (int d = 0; d < dim(); ++d)
                src_off += idx[d] * strides_[d];
            dst[i] = storage_.data_ptr()[src_off];
            for (int d = dim() - 1; d >= 0; --d) {
                if (++idx[d] < shape_[d]) break;
                idx[d] = 0;
            }
        }
    }
    return t;
}

Tensor Tensor::zeros(std::vector<int> shape) { return Tensor(std::move(shape), 0.0f); }
Tensor Tensor::ones(std::vector<int> shape) { return Tensor(std::move(shape), 1.0f); }
Tensor Tensor::empty(std::vector<int> shape) { return Tensor(std::move(shape)); }
Tensor Tensor::full(std::vector<int> shape, float value) { return Tensor(std::move(shape), value); }

Tensor Tensor::arange(float start, float end, float step) {
    if (step == 0) throw std::runtime_error("Step cannot be zero");
    int n = static_cast<int>(std::ceil((end - start) / step));
    if (n <= 0) return Tensor({0});
    Tensor t({n});
    float* p = t.data_ptr();
    for (int i = 0; i < n; ++i)
        p[i] = start + i * step;
    return t;
}

Tensor Tensor::eye(int n) {
    Tensor t = zeros({n, n});
    float* p = t.data_ptr();
    for (int i = 0; i < n; ++i)
        p[i * n + i] = 1.0f;
    return t;
}

Tensor Tensor::diag(const Tensor& tensor) {
    if (tensor.dim() != 1)
        throw std::runtime_error("diag expects a 1D tensor");
    int n = tensor.sizes()[0];
    Tensor t = zeros({n, n});
    for (int i = 0; i < n; ++i)
        t.data_ptr()[i * n + i] = tensor.at({i});
    return t;
}

// ════════════════════════════════════════════════════════════════
// Phase 3: Indexing / Access
// ════════════════════════════════════════════════════════════════

Tensor Tensor::operator[](int index) const {
    if (dim() == 0)
        throw std::runtime_error("Cannot index a 0-d tensor");
    if (index < 0 || index >= shape_[0])
        throw std::out_of_range("Index out of bounds");
    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_ + index * strides_[0];
    t.shape_ = std::vector<int>(shape_.begin() + 1, shape_.end());
    t.strides_ = std::vector<int>(strides_.begin() + 1, strides_.end());
    t.dtype_ = dtype_;
    t.device_ = device_;
    return t;
}

float Tensor::at(std::initializer_list<int> indices) const {
    return storage_.data_ptr()[flat_index(indices)];
}

void Tensor::set(std::initializer_list<int> indices, float value) {
    storage_.data_ptr()[flat_index(indices)] = value;
}

float Tensor::item() const {
    if (numel() != 1)
        throw std::runtime_error("item() requires a tensor with exactly one element");
    return data_ptr()[0];
}

// ════════════════════════════════════════════════════════════════
// Phase 4: View / Shape Ops
// ════════════════════════════════════════════════════════════════

Tensor Tensor::view(std::vector<int> shape) const {
    if (!is_contiguous())
        throw std::runtime_error("view requires a contiguous tensor");

    int inferred = -1;
    int known = 1;
    for (int i = 0; i < static_cast<int>(shape.size()); ++i) {
        if (shape[i] == -1) {
            if (inferred != -1) throw std::runtime_error("Only one -1 allowed in view");
            inferred = i;
        } else {
            known *= shape[i];
        }
    }
    if (inferred != -1) {
        if (known == 0) throw std::runtime_error("Cannot infer dimension with zero-size");
        shape[inferred] = numel() / known;
    }
    if (compute_numel(shape) != numel())
        throw std::runtime_error("view shape incompatible with tensor size");

    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_;
    t.shape_ = std::move(shape);
    t.strides_ = compute_strides(t.shape_);
    t.dtype_ = dtype_;
    t.device_ = device_;
    return t;
}

Tensor Tensor::reshape(std::vector<int> shape) const {
    if (is_contiguous())
        return view(std::move(shape));
    return clone().view(std::move(shape));
}

Tensor Tensor::flatten() const {
    return reshape({numel()});
}

Tensor Tensor::transpose(int dim0, int dim1) const {
    if (dim0 < 0 || dim0 >= dim() || dim1 < 0 || dim1 >= dim())
        throw std::out_of_range("transpose dims out of range");
    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_;
    t.shape_ = shape_;
    t.strides_ = strides_;
    t.dtype_ = dtype_;
    t.device_ = device_;
    std::swap(t.shape_[dim0], t.shape_[dim1]);
    std::swap(t.strides_[dim0], t.strides_[dim1]);
    return t;
}

Tensor Tensor::permute(std::vector<int> order) const {
    if (static_cast<int>(order.size()) != dim())
        throw std::runtime_error("permute order must match dim");
    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_;
    t.shape_.resize(dim());
    t.strides_.resize(dim());
    t.dtype_ = dtype_;
    t.device_ = device_;
    for (int i = 0; i < dim(); ++i) {
        t.shape_[i] = shape_[order[i]];
        t.strides_[i] = strides_[order[i]];
    }
    return t;
}

Tensor Tensor::squeeze(int d) const {
    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_;
    t.dtype_ = dtype_;
    t.device_ = device_;
    if (d == -1) {
        for (int i = 0; i < dim(); ++i) {
            if (shape_[i] != 1) {
                t.shape_.push_back(shape_[i]);
                t.strides_.push_back(strides_[i]);
            }
        }
    } else {
        if (d < 0 || d >= dim())
            throw std::out_of_range("squeeze dim out of range");
        for (int i = 0; i < dim(); ++i) {
            if (i == d && shape_[i] == 1) continue;
            t.shape_.push_back(shape_[i]);
            t.strides_.push_back(strides_[i]);
        }
    }
    if (t.shape_.empty() && !is_empty()) {
        // scalar tensor
    }
    return t;
}

Tensor Tensor::unsqueeze(int d) const {
    if (d < 0) d = dim() + 1 + d;
    if (d < 0 || d > dim())
        throw std::out_of_range("unsqueeze dim out of range");
    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_;
    t.shape_ = shape_;
    t.strides_ = strides_;
    t.dtype_ = dtype_;
    t.device_ = device_;
    int stride_val = (d < dim()) ? shape_[d] * strides_[d] : 1;
    if (d > 0 && d == dim()) stride_val = 1;
    t.shape_.insert(t.shape_.begin() + d, 1);
    t.strides_.insert(t.strides_.begin() + d, stride_val);
    return t;
}

Tensor Tensor::expand(std::vector<int> shape) const {
    int ndim = static_cast<int>(shape.size());
    if (ndim < dim())
        throw std::runtime_error("expand shape must have at least as many dims");

    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_;
    t.dtype_ = dtype_;
    t.device_ = device_;
    t.shape_ = shape;
    t.strides_.resize(ndim, 0);

    int offset = ndim - dim();
    for (int i = ndim - 1; i >= 0; --i) {
        int orig_i = i - offset;
        if (orig_i >= 0) {
            if (shape_[orig_i] == shape[i]) {
                t.strides_[i] = strides_[orig_i];
            } else if (shape_[orig_i] == 1) {
                t.strides_[i] = 0;
            } else {
                throw std::runtime_error("expand: incompatible shape");
            }
        } else {
            t.strides_[i] = 0;
        }
    }
    return t;
}

Tensor Tensor::narrow(int d, int start, int length) const {
    if (d < 0 || d >= dim())
        throw std::out_of_range("narrow dim out of range");
    if (start < 0 || start + length > shape_[d])
        throw std::out_of_range("narrow range out of bounds");
    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_ + start * strides_[d];
    t.shape_ = shape_;
    t.shape_[d] = length;
    t.strides_ = strides_;
    t.dtype_ = dtype_;
    t.device_ = device_;
    return t;
}

Tensor Tensor::slice(int d, int start, int end, int step) const {
    if (d < 0 || d >= dim())
        throw std::out_of_range("slice dim out of range");
    if (step <= 0) throw std::runtime_error("slice step must be positive");
    if (start < 0) start = 0;
    if (end > shape_[d]) end = shape_[d];
    int len = (end - start + step - 1) / step;
    if (len <= 0) len = 0;

    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_ + start * strides_[d];
    t.shape_ = shape_;
    t.shape_[d] = len;
    t.strides_ = strides_;
    t.strides_[d] = strides_[d] * step;
    t.dtype_ = dtype_;
    t.device_ = device_;
    return t;
}

Tensor Tensor::select(int d, int index) const {
    if (d < 0 || d >= dim())
        throw std::out_of_range("select dim out of range");
    if (index < 0 || index >= shape_[d])
        throw std::out_of_range("select index out of bounds");
    Tensor t;
    t.storage_ = storage_;
    t.offset_ = offset_ + index * strides_[d];
    t.dtype_ = dtype_;
    t.device_ = device_;
    for (int i = 0; i < dim(); ++i) {
        if (i == d) continue;
        t.shape_.push_back(shape_[i]);
        t.strides_.push_back(strides_[i]);
    }
    return t;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    return clone();
}

// ════════════════════════════════════════════════════════════════
// Phase 5: Copy / Ownership
// ════════════════════════════════════════════════════════════════

void Tensor::copy_(const Tensor& other) {
    if (numel() != other.numel())
        throw std::runtime_error("copy_ requires same number of elements");
    int n = numel();
    std::vector<int> idx(dim(), 0);
    std::vector<int> oidx(other.dim(), 0);
    for (int i = 0; i < n; ++i) {
        int dst_off = offset_;
        for (int d = 0; d < dim(); ++d)
            dst_off += idx[d] * strides_[d];
        int src_off = other.offset_;
        for (int d = 0; d < other.dim(); ++d)
            src_off += oidx[d] * other.strides_[d];
        storage_.data_ptr()[dst_off] = other.storage_.data_ptr()[src_off];

        for (int d = dim() - 1; d >= 0; --d) {
            if (++idx[d] < shape_[d]) break;
            idx[d] = 0;
        }
        for (int d = other.dim() - 1; d >= 0; --d) {
            if (++oidx[d] < other.shape_[d]) break;
            oidx[d] = 0;
        }
    }
}

Tensor Tensor::detach() const {
    Tensor t;
    t.storage_ = storage_;
    t.shape_ = shape_;
    t.strides_ = strides_;
    t.offset_ = offset_;
    t.dtype_ = dtype_;
    t.device_ = device_;
    t.requires_grad_ = false;
    return t;
}

bool Tensor::is_shared_storage() const {
    return storage_.use_count() > 1;
}

// ════════════════════════════════════════════════════════════════
// Phase 6: Elementwise Ops - Helpers
// ════════════════════════════════════════════════════════════════

std::pair<std::vector<int>, std::vector<std::pair<std::vector<int>, std::vector<int>>>>
Tensor::broadcast_shapes(const Tensor& a, const Tensor& b) {
    int ndim = std::max(a.dim(), b.dim());
    std::vector<int> result_shape(ndim);
    for (int i = 0; i < ndim; ++i) {
        int da = i - (ndim - a.dim());
        int db = i - (ndim - b.dim());
        int sa = (da >= 0) ? a.sizes()[da] : 1;
        int sb = (db >= 0) ? b.sizes()[db] : 1;
        if (sa != sb && sa != 1 && sb != 1)
            throw std::runtime_error("Shapes not broadcastable");
        result_shape[i] = std::max(sa, sb);
    }
    return {result_shape, {}};
}

Tensor Tensor::apply_unary(std::function<float(float)> fn) const {
    Tensor result(shape_);
    int n = numel();
    float* dst = result.data_ptr();

    if (is_contiguous()) {
        const float* src = data_ptr();
        for (int i = 0; i < n; ++i)
            dst[i] = fn(src[i]);
    } else {
        std::vector<int> idx(dim(), 0);
        for (int i = 0; i < n; ++i) {
            int src_off = offset_;
            for (int d = 0; d < dim(); ++d)
                src_off += idx[d] * strides_[d];
            dst[i] = fn(storage_.data_ptr()[src_off]);
            for (int d = dim() - 1; d >= 0; --d) {
                if (++idx[d] < shape_[d]) break;
                idx[d] = 0;
            }
        }
    }
    return result;
}

Tensor Tensor::apply_binary(const Tensor& other, std::function<float(float, float)> fn) const {
    auto [result_shape, _] = broadcast_shapes(*this, other);
    int ndim = static_cast<int>(result_shape.size());
    Tensor result(result_shape);
    int n = result.numel();
    float* dst = result.data_ptr();

    std::vector<int> a_strides(ndim, 0);
    std::vector<int> b_strides(ndim, 0);
    for (int i = 0; i < ndim; ++i) {
        int da = i - (ndim - dim());
        int db = i - (ndim - other.dim());
        if (da >= 0 && shape_[da] > 1) a_strides[i] = strides_[da];
        if (db >= 0 && other.shape_[db] > 1) b_strides[i] = other.strides_[db];
    }

    std::vector<int> idx(ndim, 0);
    for (int i = 0; i < n; ++i) {
        int a_off = offset_, b_off = other.offset_;
        for (int d = 0; d < ndim; ++d) {
            a_off += idx[d] * a_strides[d];
            b_off += idx[d] * b_strides[d];
        }
        dst[i] = fn(storage_.data_ptr()[a_off], other.storage_.data_ptr()[b_off]);
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < result_shape[d]) break;
            idx[d] = 0;
        }
    }
    return result;
}

Tensor& Tensor::apply_binary_inplace(const Tensor& other, std::function<float(float, float)> fn) {
    int n = numel();
    int ndim = dim();
    std::vector<int> b_strides(ndim, 0);
    for (int i = 0; i < ndim; ++i) {
        int db = i - (ndim - other.dim());
        if (db >= 0 && other.shape_[db] > 1) b_strides[i] = other.strides_[db];
    }

    std::vector<int> idx(ndim, 0);
    for (int i = 0; i < n; ++i) {
        int a_off = offset_, b_off = other.offset_;
        for (int d = 0; d < ndim; ++d) {
            a_off += idx[d] * strides_[d];
            b_off += idx[d] * b_strides[d];
        }
        storage_.data_ptr()[a_off] = fn(storage_.data_ptr()[a_off], other.storage_.data_ptr()[b_off]);
        for (int d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < shape_[d]) break;
            idx[d] = 0;
        }
    }
    return *this;
}

// Phase 6: Elementwise Ops - Tensor-Tensor
Tensor Tensor::add(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a + b; }); }
Tensor Tensor::sub(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a - b; }); }
Tensor Tensor::mul(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a * b; }); }
Tensor Tensor::div(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a / b; }); }
Tensor Tensor::pow(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return std::pow(a, b); }); }

// Phase 6: Unary
Tensor Tensor::neg() const     { return apply_unary([](float x) { return -x; }); }
Tensor Tensor::exp() const     { return apply_unary([](float x) { return std::exp(x); }); }
Tensor Tensor::log() const     { return apply_unary([](float x) { return std::log(x); }); }
Tensor Tensor::sqrt() const    { return apply_unary([](float x) { return std::sqrt(x); }); }
Tensor Tensor::abs() const     { return apply_unary([](float x) { return std::abs(x); }); }
Tensor Tensor::relu() const    { return apply_unary([](float x) { return x > 0 ? x : 0.0f; }); }
Tensor Tensor::sigmoid() const { return apply_unary([](float x) { return 1.0f / (1.0f + std::exp(-x)); }); }
Tensor Tensor::tanh() const    { return apply_unary([](float x) { return std::tanh(x); }); }

// Phase 6: Scalar ops
Tensor Tensor::add_scalar(float v) const { return apply_unary([v](float x) { return x + v; }); }
Tensor Tensor::sub_scalar(float v) const { return apply_unary([v](float x) { return x - v; }); }
Tensor Tensor::mul_scalar(float v) const { return apply_unary([v](float x) { return x * v; }); }
Tensor Tensor::div_scalar(float v) const { return apply_unary([v](float x) { return x / v; }); }
Tensor Tensor::pow_scalar(float v) const { return apply_unary([v](float x) { return std::pow(x, v); }); }

// Phase 6: In-place
Tensor& Tensor::add_(const Tensor& o) { return apply_binary_inplace(o, [](float a, float b) { return a + b; }); }
Tensor& Tensor::sub_(const Tensor& o) { return apply_binary_inplace(o, [](float a, float b) { return a - b; }); }
Tensor& Tensor::mul_(const Tensor& o) { return apply_binary_inplace(o, [](float a, float b) { return a * b; }); }
Tensor& Tensor::div_(const Tensor& o) { return apply_binary_inplace(o, [](float a, float b) { return a / b; }); }
Tensor& Tensor::add_scalar_(float v) {
    int n = numel();
    std::vector<int> idx(dim(), 0);
    for (int i = 0; i < n; ++i) {
        int off = offset_;
        for (int d = 0; d < dim(); ++d) off += idx[d] * strides_[d];
        storage_.data_ptr()[off] += v;
        for (int d = dim() - 1; d >= 0; --d) { if (++idx[d] < shape_[d]) break; idx[d] = 0; }
    }
    return *this;
}
Tensor& Tensor::mul_scalar_(float v) {
    int n = numel();
    std::vector<int> idx(dim(), 0);
    for (int i = 0; i < n; ++i) {
        int off = offset_;
        for (int d = 0; d < dim(); ++d) off += idx[d] * strides_[d];
        storage_.data_ptr()[off] *= v;
        for (int d = dim() - 1; d >= 0; --d) { if (++idx[d] < shape_[d]) break; idx[d] = 0; }
    }
    return *this;
}

// ════════════════════════════════════════════════════════════════
// Phase 7: Reduction Ops
// ════════════════════════════════════════════════════════════════

Tensor Tensor::reduce(int d, std::function<float(float, float)> fn, float init) const {
    if (d == -1) {
        float result = init;
        Tensor flat = contiguous();
        const float* p = flat.data_ptr();
        for (int i = 0; i < numel(); ++i)
            result = fn(result, p[i]);
        return Tensor({1}, result);
    }
    if (d < 0 || d >= dim())
        throw std::out_of_range("reduce dim out of range");

    std::vector<int> out_shape;
    for (int i = 0; i < dim(); ++i)
        if (i != d) out_shape.push_back(shape_[i]);
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, init);
    int n = numel();
    std::vector<int> idx(dim(), 0);
    for (int i = 0; i < n; ++i) {
        int src_off = offset_;
        for (int dd = 0; dd < dim(); ++dd)
            src_off += idx[dd] * strides_[dd];

        std::vector<int> out_idx;
        for (int dd = 0; dd < dim(); ++dd)
            if (dd != d) out_idx.push_back(idx[dd]);
        if (out_idx.empty()) out_idx.push_back(0);

        int dst_off = 0;
        for (int dd = 0; dd < static_cast<int>(out_idx.size()); ++dd)
            dst_off += out_idx[dd] * result.strides_[dd];

        result.storage_.data_ptr()[dst_off] =
            fn(result.storage_.data_ptr()[dst_off], storage_.data_ptr()[src_off]);

        for (int dd = dim() - 1; dd >= 0; --dd) {
            if (++idx[dd] < shape_[dd]) break;
            idx[dd] = 0;
        }
    }
    return result;
}

Tensor Tensor::reduce_arg(int d, std::function<bool(float, float)> cmp) const {
    if (d == -1) {
        Tensor flat = contiguous();
        const float* p = flat.data_ptr();
        int best = 0;
        for (int i = 1; i < numel(); ++i)
            if (cmp(p[i], p[best])) best = i;
        return Tensor({1}, static_cast<float>(best));
    }
    if (d < 0 || d >= dim())
        throw std::out_of_range("reduce_arg dim out of range");

    std::vector<int> out_shape;
    for (int i = 0; i < dim(); ++i)
        if (i != d) out_shape.push_back(shape_[i]);
    if (out_shape.empty()) out_shape.push_back(1);

    Tensor result(out_shape, 0.0f);
    Tensor best_vals(out_shape, 0.0f);
    bool first_pass = true;

    int n = numel();
    std::vector<int> idx(dim(), 0);
    for (int i = 0; i < n; ++i) {
        int src_off = offset_;
        for (int dd = 0; dd < dim(); ++dd)
            src_off += idx[dd] * strides_[dd];
        float val = storage_.data_ptr()[src_off];

        std::vector<int> out_idx;
        for (int dd = 0; dd < dim(); ++dd)
            if (dd != d) out_idx.push_back(idx[dd]);
        if (out_idx.empty()) out_idx.push_back(0);

        int dst_off = 0;
        for (int dd = 0; dd < static_cast<int>(out_idx.size()); ++dd)
            dst_off += out_idx[dd] * result.strides_[dd];

        if (idx[d] == 0) {
            best_vals.storage_.data_ptr()[dst_off] = val;
            result.storage_.data_ptr()[dst_off] = 0.0f;
        } else if (cmp(val, best_vals.storage_.data_ptr()[dst_off])) {
            best_vals.storage_.data_ptr()[dst_off] = val;
            result.storage_.data_ptr()[dst_off] = static_cast<float>(idx[d]);
        }

        for (int dd = dim() - 1; dd >= 0; --dd) {
            if (++idx[dd] < shape_[dd]) break;
            idx[dd] = 0;
        }
    }
    return result;
}

Tensor Tensor::sum(int d) const {
    return reduce(d, [](float a, float b) { return a + b; }, 0.0f);
}

Tensor Tensor::mean(int d) const {
    Tensor s = sum(d);
    float count = (d == -1) ? static_cast<float>(numel()) : static_cast<float>(shape_[d]);
    return s.div_scalar(count);
}

Tensor Tensor::max(int d) const {
    return reduce(d, [](float a, float b) { return a > b ? a : b; }, -std::numeric_limits<float>::infinity());
}

Tensor Tensor::min(int d) const {
    return reduce(d, [](float a, float b) { return a < b ? a : b; }, std::numeric_limits<float>::infinity());
}

Tensor Tensor::argmax(int d) const {
    return reduce_arg(d, [](float a, float b) { return a > b; });
}

Tensor Tensor::argmin(int d) const {
    return reduce_arg(d, [](float a, float b) { return a < b; });
}

// ════════════════════════════════════════════════════════════════
// Phase 8: Comparison Ops
// ════════════════════════════════════════════════════════════════

Tensor Tensor::eq(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a == b ? 1.0f : 0.0f; }); }
Tensor Tensor::ne(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a != b ? 1.0f : 0.0f; }); }
Tensor Tensor::lt(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a <  b ? 1.0f : 0.0f; }); }
Tensor Tensor::le(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a <= b ? 1.0f : 0.0f; }); }
Tensor Tensor::gt(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a >  b ? 1.0f : 0.0f; }); }
Tensor Tensor::ge(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return a >= b ? 1.0f : 0.0f; }); }

// ════════════════════════════════════════════════════════════════
// Phase 8: Linear Algebra
// ════════════════════════════════════════════════════════════════

Tensor Tensor::matmul(const Tensor& other) const {
    if (dim() == 1 && other.dim() == 1)
        return dot(other);
    if (dim() == 2 && other.dim() == 2)
        return mm(other);
    if (dim() == 3 && other.dim() == 3)
        return bmm(other);
    if (dim() == 2 && other.dim() == 1) {
        Tensor b2 = other.unsqueeze(1);
        Tensor r = mm(b2);
        return r.squeeze(1);
    }
    if (dim() == 1 && other.dim() == 2) {
        Tensor a2 = unsqueeze(0);
        Tensor r = a2.mm(other);
        return r.squeeze(0);
    }
    throw std::runtime_error("matmul: unsupported dimensions");
}

Tensor Tensor::mm(const Tensor& other) const {
    if (dim() != 2 || other.dim() != 2)
        throw std::runtime_error("mm requires 2D tensors");
    if (shape_[1] != other.shape_[0])
        throw std::runtime_error("mm: incompatible shapes");

    int M = shape_[0], K = shape_[1], N = other.shape_[1];
    Tensor result = zeros({M, N});
    Tensor a_c = contiguous();
    Tensor b_c = other.contiguous();
    const float* A = a_c.data_ptr();
    const float* B = b_c.data_ptr();
    float* C = result.data_ptr();

    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < N; ++j)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
    return result;
}

Tensor Tensor::bmm(const Tensor& other) const {
    if (dim() != 3 || other.dim() != 3)
        throw std::runtime_error("bmm requires 3D tensors");
    if (shape_[0] != other.shape_[0] || shape_[2] != other.shape_[1])
        throw std::runtime_error("bmm: incompatible shapes");

    int batch = shape_[0], M = shape_[1], K = shape_[2], N = other.shape_[2];
    Tensor result = zeros({batch, M, N});
    Tensor a_c = contiguous();
    Tensor b_c = other.contiguous();

    for (int b = 0; b < batch; ++b) {
        const float* A = a_c.data_ptr() + b * M * K;
        const float* B = b_c.data_ptr() + b * K * N;
        float* C = result.data_ptr() + b * M * N;
        for (int i = 0; i < M; ++i)
            for (int k = 0; k < K; ++k)
                for (int j = 0; j < N; ++j)
                    C[i * N + j] += A[i * K + k] * B[k * N + j];
    }
    return result;
}

Tensor Tensor::dot(const Tensor& other) const {
    if (dim() != 1 || other.dim() != 1)
        throw std::runtime_error("dot requires 1D tensors");
    if (shape_[0] != other.shape_[0])
        throw std::runtime_error("dot: size mismatch");
    Tensor a_c = contiguous();
    Tensor b_c = other.contiguous();
    float result = 0;
    const float* a = a_c.data_ptr();
    const float* b = b_c.data_ptr();
    for (int i = 0; i < shape_[0]; ++i)
        result += a[i] * b[i];
    return Tensor({1}, result);
}

Tensor Tensor::outer(const Tensor& other) const {
    if (dim() != 1 || other.dim() != 1)
        throw std::runtime_error("outer requires 1D tensors");
    int M = shape_[0], N = other.shape_[0];
    Tensor result({M, N});
    Tensor a_c = contiguous();
    Tensor b_c = other.contiguous();
    const float* a = a_c.data_ptr();
    const float* b = b_c.data_ptr();
    float* r = result.data_ptr();
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            r[i * N + j] = a[i] * b[j];
    return result;
}

// ════════════════════════════════════════════════════════════════
// Phase 8: Concatenation / Splitting
// ════════════════════════════════════════════════════════════════

Tensor Tensor::cat(const std::vector<Tensor>& tensors, int d) {
    if (tensors.empty()) throw std::runtime_error("cat: empty tensor list");
    int ndim = tensors[0].dim();
    if (d < 0) d += ndim;
    if (d < 0 || d >= ndim)
        throw std::out_of_range("cat: dim out of range");

    std::vector<int> result_shape = tensors[0].sizes();
    for (size_t t = 1; t < tensors.size(); ++t) {
        if (tensors[t].dim() != ndim)
            throw std::runtime_error("cat: dimension mismatch");
        for (int i = 0; i < ndim; ++i) {
            if (i == d) {
                result_shape[i] += tensors[t].sizes()[i];
            } else if (tensors[t].sizes()[i] != result_shape[i]) {
                throw std::runtime_error("cat: shape mismatch");
            }
        }
    }

    Tensor result(result_shape);
    int offset = 0;
    for (const auto& t : tensors) {
        Tensor dst = result.narrow(d, offset, t.sizes()[d]);
        dst.copy_(t);
        offset += t.sizes()[d];
    }
    return result;
}

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int d) {
    if (tensors.empty()) throw std::runtime_error("stack: empty tensor list");
    std::vector<Tensor> unsqueezed;
    unsqueezed.reserve(tensors.size());
    for (const auto& t : tensors)
        unsqueezed.push_back(t.unsqueeze(d));
    return cat(unsqueezed, d);
}

std::vector<Tensor> Tensor::split(int size, int d) const {
    if (d < 0) d += dim();
    if (d < 0 || d >= dim())
        throw std::out_of_range("split: dim out of range");
    std::vector<Tensor> result;
    int total = shape_[d];
    for (int start = 0; start < total; start += size) {
        int len = std::min(size, total - start);
        result.push_back(narrow(d, start, len));
    }
    return result;
}

std::vector<Tensor> Tensor::chunk(int chunks, int d) const {
    if (d < 0) d += dim();
    if (d < 0 || d >= dim())
        throw std::out_of_range("chunk: dim out of range");
    int size = (shape_[d] + chunks - 1) / chunks;
    return split(size, d);
}

// ════════════════════════════════════════════════════════════════
// Phase 9: Autograd
// ════════════════════════════════════════════════════════════════

bool Tensor::requires_grad() const { return requires_grad_; }
void Tensor::set_requires_grad(bool req) { requires_grad_ = req; }

Tensor Tensor::grad() const {
    if (!grad_) return Tensor();
    return *grad_;
}

void Tensor::set_grad(const Tensor& g) {
    grad_ = std::make_shared<Tensor>(g.clone());
}

std::shared_ptr<GradFunction> Tensor::grad_fn() const { return grad_fn_; }

void Tensor::backward() {
    backward(ones(shape_));
}

void Tensor::backward(const Tensor& grad_output) {
    if (!requires_grad_)
        throw std::runtime_error("backward called on tensor that doesn't require grad");
    if (!grad_) {
        grad_ = std::make_shared<Tensor>(grad_output.clone());
    } else {
        grad_->add_(grad_output);
    }
}

void Tensor::zero_grad() {
    if (grad_)
        grad_.reset();
}

// ════════════════════════════════════════════════════════════════
// Phase 10: Utility / Debug
// ════════════════════════════════════════════════════════════════

std::string Tensor::shape_string() const {
    std::ostringstream ss;
    ss << "(";
    for (int i = 0; i < dim(); ++i) {
        if (i > 0) ss << ", ";
        ss << shape_[i];
    }
    ss << ")";
    return ss.str();
}

std::string Tensor::stride_string() const {
    std::ostringstream ss;
    ss << "(";
    for (int i = 0; i < dim(); ++i) {
        if (i > 0) ss << ", ";
        ss << strides_[i];
    }
    ss << ")";
    return ss.str();
}

static void print_recursive(std::ostream& os, const Tensor& t, std::vector<int>& idx, int d) {
    if (d == t.dim()) {
        int off = t.storage_offset();
        for (int i = 0; i < t.dim(); ++i)
            off += idx[i] * t.strides()[i];
        os << t.storage().data_ptr()[off];
        return;
    }
    os << "[";
    for (int i = 0; i < t.sizes()[d]; ++i) {
        if (i > 0) os << ", ";
        idx[d] = i;
        print_recursive(os, t, idx, d + 1);
    }
    os << "]";
}

std::string Tensor::to_string() const {
    if (is_empty()) return "Tensor([])";
    if (dim() == 0) return "Tensor(" + std::to_string(data_ptr()[0]) + ")";
    std::ostringstream ss;
    ss << "Tensor(";
    std::vector<int> idx(dim(), 0);
    print_recursive(ss, *this, idx, 0);
    ss << ", shape=" << shape_string();
    ss << ", dtype=" << dtype_name(dtype_);
    ss << ")";
    return ss.str();
}

void Tensor::print(std::ostream& os) const {
    os << to_string() << std::endl;
}

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
    os << t.to_string();
    return os;
}

Tensor Tensor::to(Device dev) const {
    Tensor t = clone();
    t.device_ = dev;
    return t;
}

Tensor Tensor::astype(DType dt) const {
    Tensor t = clone();
    t.dtype_ = dt;
    return t;
}

Tensor Tensor::pin_memory() const {
    return clone();
}

} // namespace minitorch
