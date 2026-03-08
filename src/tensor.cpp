#include <minitorch/tensor.hpp>
#include <minitorch/autograd.hpp>

namespace minitorch {

thread_local std::mt19937 Tensor::rng_(std::random_device{}());

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

static bool any_requires_grad(const Tensor& a) {
    return a.requires_grad();
}

static bool any_requires_grad(const Tensor& a, const Tensor& b) {
    return a.requires_grad() || b.requires_grad();
}

Edge make_edge(const Tensor& t) {
    if (t.grad_fn()) {
        return {t.grad_fn(), 0};
    }
    if (t.requires_grad()) {
        if (!t.accumulate_grad_) {
            if (!t.grad_holder_)
                const_cast<Tensor&>(t).grad_holder_ = std::make_shared<GradHolder>();
            auto accum = std::make_shared<AccumulateGrad>();
            accum->holder = t.grad_holder_;
            const_cast<Tensor&>(t).accumulate_grad_ = accum;
        }
        return {t.accumulate_grad_, 0};
    }
    return {nullptr, 0};
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

Tensor Tensor::randn(std::vector<int> shape) {
    Tensor t(shape);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float* p = t.data_ptr();
    int n = t.numel();
    for (int i = 0; i < n; ++i)
        p[i] = dist(rng_);
    return t;
}

Tensor Tensor::tril(int n, int diagonal) {
    Tensor t = zeros({n, n});
    float* p = t.data_ptr();
    for (int i = 0; i < n; ++i)
        for (int j = 0; j <= i + diagonal && j < n; ++j)
            if (j >= 0) p[i * n + j] = 1.0f;
    return t;
}

Tensor Tensor::triu(int n, int diagonal) {
    Tensor t = zeros({n, n});
    float* p = t.data_ptr();
    for (int i = 0; i < n; ++i)
        for (int j = std::max(0, i + diagonal); j < n; ++j)
            p[i * n + j] = 1.0f;
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
    t.shape_ = shape;
    t.strides_ = compute_strides(t.shape_);
    t.dtype_ = dtype_;
    t.device_ = device_;

    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<ReshapeBackward>();
        fn->self_shape = shape_;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
    }
    return t;
}

Tensor Tensor::reshape(std::vector<int> shape) const {
    if (is_contiguous()) {
        return view(std::move(shape));
    }
    Tensor t = clone();
    t.shape_ = std::move(shape);
    t.strides_ = compute_strides(t.shape_);
    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<ReshapeBackward>();
        fn->self_shape = shape_;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
    }
    return t;
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

    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<TransposeBackward>();
        fn->dim0 = dim0;
        fn->dim1 = dim1;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
    }
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

    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<PermuteBackward>();
        fn->order = order;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
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

    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<SqueezeBackward>();
        fn->self_shape = shape_;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
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

    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<UnsqueezeBackward>();
        fn->dim = d;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
    }
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

    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<ExpandBackward>();
        fn->self_shape = shape_;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
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

    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<SliceBackward>();
        fn->self_shape = shape_;
        fn->dim = d;
        fn->start = start;
        fn->end = end;
        fn->step = step;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
    }
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

    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<SelectBackward>();
        fn->self_shape = shape_;
        fn->dim = d;
        fn->index = index;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
    }
    return t;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) return *this;
    Tensor t = clone();
    if (requires_grad_) {
        t.requires_grad_ = true;
        auto fn = std::make_shared<ReshapeBackward>();
        fn->self_shape = shape_;
        fn->add_next_edge(make_edge(*this));
        t.grad_fn_ = fn;
    }
    return t;
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

// Phase 6: Elementwise Ops with autograd

Tensor Tensor::add(const Tensor& o) const {
    Tensor result = apply_binary(o, [](float a, float b) { return a + b; });
    if (any_requires_grad(*this, o)) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<AddBackward>();
        fn->self_shape = shape_;
        fn->other_shape = o.shape_;
        fn->add_next_edge(make_edge(*this));
        fn->add_next_edge(make_edge(o));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::sub(const Tensor& o) const {
    Tensor result = apply_binary(o, [](float a, float b) { return a - b; });
    if (any_requires_grad(*this, o)) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<SubBackward>();
        fn->self_shape = shape_;
        fn->other_shape = o.shape_;
        fn->add_next_edge(make_edge(*this));
        fn->add_next_edge(make_edge(o));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::mul(const Tensor& o) const {
    Tensor result = apply_binary(o, [](float a, float b) { return a * b; });
    if (any_requires_grad(*this, o)) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<MulBackward>();
        fn->self = detach();
        fn->other = o.detach();
        fn->add_next_edge(make_edge(*this));
        fn->add_next_edge(make_edge(o));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::div(const Tensor& o) const {
    Tensor result = apply_binary(o, [](float a, float b) { return a / b; });
    if (any_requires_grad(*this, o)) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<DivBackward>();
        fn->self = detach();
        fn->other = o.detach();
        fn->add_next_edge(make_edge(*this));
        fn->add_next_edge(make_edge(o));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::pow(const Tensor& o) const { return apply_binary(o, [](float a, float b) { return std::pow(a, b); }); }

Tensor Tensor::neg() const {
    Tensor result = apply_unary([](float x) { return -x; });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<NegBackward>();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::exp() const {
    Tensor result = apply_unary([](float x) { return std::exp(x); });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<ExpBackward>();
        fn->result = result.detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::log() const {
    Tensor result = apply_unary([](float x) { return std::log(x); });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<LogBackward>();
        fn->self = detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::sqrt() const {
    Tensor result = apply_unary([](float x) { return std::sqrt(x); });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<SqrtBackward>();
        fn->result = result.detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::abs() const {
    Tensor result = apply_unary([](float x) { return std::abs(x); });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<AbsBackward>();
        fn->self = detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::relu() const {
    Tensor result = apply_unary([](float x) { return x > 0 ? x : 0.0f; });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<ReluBackward>();
        fn->self = detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::sigmoid() const {
    Tensor result = apply_unary([](float x) { return 1.0f / (1.0f + std::exp(-x)); });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<SigmoidBackward>();
        fn->result = result.detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::tanh() const {
    Tensor result = apply_unary([](float x) { return std::tanh(x); });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<TanhBackward>();
        fn->result = result.detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::gelu() const {
    static const float kSqrt2OverPi = std::sqrt(2.0f / 3.14159265358979323846f);
    Tensor result = apply_unary([](float x) {
        float inner = std::sqrt(2.0f / 3.14159265358979323846f) * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1.0f + std::tanh(inner));
    });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<GELUBackward>();
        fn->self = detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::add_scalar(float v) const {
    Tensor result = apply_unary([v](float x) { return x + v; });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<AddScalarBackward>();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::sub_scalar(float v) const {
    Tensor result = apply_unary([v](float x) { return x - v; });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<SubScalarBackward>();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::mul_scalar(float v) const {
    Tensor result = apply_unary([v](float x) { return x * v; });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<MulScalarBackward>();
        fn->scalar = v;
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::div_scalar(float v) const {
    Tensor result = apply_unary([v](float x) { return x / v; });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<DivScalarBackward>();
        fn->scalar = v;
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::pow_scalar(float v) const {
    Tensor result = apply_unary([v](float x) { return std::pow(x, v); });
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<PowScalarBackward>();
        fn->self = detach();
        fn->scalar = v;
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

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
    Tensor result = reduce(d, [](float a, float b) { return a + b; }, 0.0f);
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<SumBackward>();
        fn->self_shape = shape_;
        fn->dim = d;
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::mean(int d) const {
    Tensor s = sum(d);
    float count = (d == -1) ? static_cast<float>(numel()) : static_cast<float>(shape_[d]);
    Tensor result = s.detach().div_scalar(count);
    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<MeanBackward>();
        fn->self_shape = shape_;
        fn->dim = d;
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
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

Tensor Tensor::variance(int d, bool unbiased) const {
    Tensor m = mean(d);
    Tensor diff;
    if (d == -1) {
        diff = sub(m.expand(shape_));
    } else {
        std::vector<int> expand_shape = shape_;
        expand_shape[d] = 1;
        Tensor m_unsq = m.reshape(expand_shape).expand(shape_);
        diff = sub(m_unsq);
    }
    Tensor sq = diff.mul(diff);
    Tensor s = sq.sum(d);
    float count = (d == -1) ? static_cast<float>(numel()) : static_cast<float>(shape_[d]);
    if (unbiased && count > 1) count -= 1.0f;
    return s.detach().div_scalar(count);
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
// Phase 8: Linear Algebra (with autograd)
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

    if (any_requires_grad(*this, other)) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<MmBackward>();
        fn->self = a_c.detach();
        fn->other = b_c.detach();
        fn->add_next_edge(make_edge(*this));
        fn->add_next_edge(make_edge(other));
        result.grad_fn_ = fn;
    }
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

    if (any_requires_grad(*this, other)) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<BmmBackward>();
        fn->self = a_c.detach();
        fn->other = b_c.detach();
        fn->add_next_edge(make_edge(*this));
        fn->add_next_edge(make_edge(other));
        result.grad_fn_ = fn;
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

    bool need_grad = false;
    for (const auto& t : tensors)
        if (t.requires_grad()) { need_grad = true; break; }
    if (need_grad) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<CatBackward>();
        fn->dim = d;
        for (const auto& t : tensors) {
            fn->split_sizes.push_back(t.sizes()[d]);
            fn->add_next_edge(make_edge(t));
        }
        result.grad_fn_ = fn;
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
// New ops for GPT
// ════════════════════════════════════════════════════════════════

Tensor Tensor::softmax(int d) const {
    if (d < 0) d += dim();
    Tensor m = max(d);
    std::vector<int> expand_shape = shape_;
    expand_shape[d] = 1;
    Tensor shifted = sub(m.reshape(expand_shape).expand(shape_));
    Tensor e = shifted.detach().apply_unary([](float x) { return std::exp(x); });
    Tensor s = e.sum(d);
    Tensor result = e.apply_binary(s.reshape(expand_shape).expand(shape_),
        [](float a, float b) { return a / b; });

    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<SoftmaxBackward>();
        fn->result = result.detach();
        fn->dim = d;
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::log_softmax(int d) const {
    if (d < 0) d += dim();
    Tensor m = max(d);
    std::vector<int> expand_shape = shape_;
    expand_shape[d] = 1;
    Tensor shifted = sub(m.reshape(expand_shape).expand(shape_)).detach();
    Tensor e = shifted.apply_unary([](float x) { return std::exp(x); });
    Tensor s = e.sum(d);
    Tensor log_s = s.apply_unary([](float x) { return std::log(x); });
    Tensor result = shifted.sub(log_s.reshape(expand_shape).expand(shape_));

    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<LogSoftmaxBackward>();
        fn->result = result.detach();
        fn->dim = d;
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::masked_fill(const Tensor& mask, float value) const {
    Tensor result = clone();
    int n = result.numel();
    float* dst = result.data_ptr();
    Tensor mc = mask.contiguous();
    const float* mp = mc.data_ptr();
    Tensor sc = contiguous();
    const float* sp = sc.data_ptr();
    for (int i = 0; i < n; ++i)
        dst[i] = (mp[i] != 0.0f) ? value : sp[i];

    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<MaskedFillBackward>();
        fn->mask = mask.detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::cross_entropy_loss(const Tensor& targets) const {
    int N = shape_[0];
    int C = shape_[1];
    Tensor lsm = log_softmax(1).detach();

    float loss = 0.0f;
    Tensor tc = targets.contiguous();
    const float* tp = tc.data_ptr();
    for (int i = 0; i < N; ++i) {
        int cls = static_cast<int>(tp[i]);
        loss -= lsm.at({i, cls});
    }
    loss /= static_cast<float>(N);
    Tensor result({1}, loss);

    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<CrossEntropyBackward>();
        fn->log_probs = lsm;
        fn->targets = targets.detach();
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::embedding_lookup(const Tensor& indices) const {
    int embed_dim = shape_[1];
    std::vector<int> out_shape = indices.sizes();
    out_shape.push_back(embed_dim);

    Tensor result(out_shape);
    Tensor ic = indices.contiguous();
    const float* ip = ic.data_ptr();
    Tensor wc = contiguous();
    const float* wp = wc.data_ptr();
    float* rp = result.data_ptr();

    int n_indices = indices.numel();
    for (int i = 0; i < n_indices; ++i) {
        int idx = static_cast<int>(ip[i]);
        std::memcpy(rp + i * embed_dim, wp + idx * embed_dim, embed_dim * sizeof(float));
    }

    if (requires_grad_) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<EmbeddingBackward>();
        fn->indices = indices.detach();
        fn->num_embeddings = shape_[0];
        fn->add_next_edge(make_edge(*this));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::layer_norm(const Tensor& weight, const Tensor& bias, float eps) const {
    int norm_dim = shape_.back();
    int outer = numel() / norm_dim;

    Tensor input_c = contiguous();
    Tensor result(shape_);
    Tensor mean_t({outer});
    Tensor rstd_t({outer});

    const float* inp = input_c.data_ptr();
    float* out = result.data_ptr();
    float* mp = mean_t.data_ptr();
    float* rp = rstd_t.data_ptr();

    Tensor wc = weight.contiguous();
    Tensor bc = bias.contiguous();
    const float* wp = wc.data_ptr();
    const float* bp = bc.data_ptr();

    for (int i = 0; i < outer; ++i) {
        const float* row = inp + i * norm_dim;
        float m = 0.0f;
        for (int j = 0; j < norm_dim; ++j) m += row[j];
        m /= norm_dim;
        mp[i] = m;

        float var = 0.0f;
        for (int j = 0; j < norm_dim; ++j) {
            float d = row[j] - m;
            var += d * d;
        }
        var /= norm_dim;
        float rs = 1.0f / std::sqrt(var + eps);
        rp[i] = rs;

        float* orow = out + i * norm_dim;
        for (int j = 0; j < norm_dim; ++j)
            orow[j] = (row[j] - m) * rs * wp[j] + bp[j];
    }

    if (any_requires_grad(*this, weight) || bias.requires_grad()) {
        result.requires_grad_ = true;
        auto fn = std::make_shared<LayerNormBackward>();
        fn->self = input_c.detach();
        fn->mean = mean_t;
        fn->rstd = rstd_t;
        fn->weight = weight.detach();
        fn->normalized_dim = norm_dim;
        fn->add_next_edge(make_edge(*this));
        fn->add_next_edge(make_edge(weight));
        fn->add_next_edge(make_edge(bias));
        result.grad_fn_ = fn;
    }
    return result;
}

Tensor Tensor::dropout(float p, bool training) const {
    if (!training || p == 0.0f) return *this;
    Tensor mask(shape_);
    int n = numel();
    float* mp = mask.data_ptr();
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float scale = 1.0f / (1.0f - p);
    for (int i = 0; i < n; ++i)
        mp[i] = (dist(rng_) >= p) ? scale : 0.0f;
    return mul(mask);
}

// ════════════════════════════════════════════════════════════════
// Phase 9: Autograd
// ════════════════════════════════════════════════════════════════

bool Tensor::requires_grad() const { return requires_grad_; }
void Tensor::set_requires_grad(bool req) {
    requires_grad_ = req;
    if (req && !grad_holder_)
        grad_holder_ = std::make_shared<GradHolder>();
}

Tensor Tensor::grad() const {
    if (!grad_holder_ || !grad_holder_->grad) return Tensor();
    return *grad_holder_->grad;
}

void Tensor::set_grad(const Tensor& g) {
    if (!grad_holder_) grad_holder_ = std::make_shared<GradHolder>();
    grad_holder_->grad = std::make_shared<Tensor>(g.clone());
}

std::shared_ptr<GradFunction> Tensor::grad_fn() const { return grad_fn_; }
void Tensor::set_grad_fn(std::shared_ptr<GradFunction> fn) { grad_fn_ = std::move(fn); }

void Tensor::backward() {
    backward(ones(shape_));
}

void Tensor::backward(const Tensor& grad_output) {
    if (!requires_grad_)
        throw std::runtime_error("backward called on tensor that doesn't require grad");

    std::vector<std::shared_ptr<GradFunction>> topo_order;
    std::unordered_set<GradFunction*> visited;

    // Iterative topological sort (post-order DFS)
    {
        struct Frame {
            std::shared_ptr<GradFunction> fn;
            size_t child_idx;
        };
        std::vector<Frame> stack;
        if (grad_fn_ && !visited.count(grad_fn_.get())) {
            visited.insert(grad_fn_.get());
            stack.push_back({grad_fn_, 0});
        }
        while (!stack.empty()) {
            auto& top = stack.back();
            if (top.child_idx < top.fn->next_edges.size()) {
                auto& edge = top.fn->next_edges[top.child_idx++];
                if (edge.function && !visited.count(edge.function.get())) {
                    visited.insert(edge.function.get());
                    stack.push_back({edge.function, 0});
                }
            } else {
                topo_order.push_back(top.fn);
                stack.pop_back();
            }
        }
    }

    std::unordered_map<GradFunction*, Tensor> grad_map;
    if (grad_fn_) {
        grad_map[grad_fn_.get()] = grad_output.clone();
    } else {
        if (!grad_holder_) grad_holder_ = std::make_shared<GradHolder>();
        if (!grad_holder_->grad) {
            grad_holder_->grad = std::make_shared<Tensor>(grad_output.clone());
        } else {
            grad_holder_->grad->add_(grad_output);
        }
        return;
    }

    for (int i = static_cast<int>(topo_order.size()) - 1; i >= 0; --i) {
        auto& fn = topo_order[i];
        auto it = grad_map.find(fn.get());
        if (it == grad_map.end()) continue;

        Tensor current_grad = it->second;
        std::vector<Tensor> input_grads = fn->apply({current_grad});

        for (size_t j = 0; j < fn->next_edges.size() && j < input_grads.size(); ++j) {
            auto& edge = fn->next_edges[j];
            if (!edge.function) continue;

            auto accum = std::dynamic_pointer_cast<AccumulateGrad>(edge.function);
            if (accum) {
                accum->apply({input_grads[j]});
                continue;
            }

            auto git = grad_map.find(edge.function.get());
            if (git == grad_map.end()) {
                grad_map[edge.function.get()] = input_grads[j].clone();
            } else {
                git->second.add_(input_grads[j]);
            }
        }
    }
}

void Tensor::zero_grad() {
    if (grad_holder_ && grad_holder_->grad)
        grad_holder_->grad.reset();
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

// ════════════════════════════════════════════════════════════════
// Autograd backward implementations
// ════════════════════════════════════════════════════════════════

Tensor reduce_grad_for_broadcast(const Tensor& grad, const std::vector<int>& target_shape) {
    Tensor g = grad.detach();
    int target_dim = static_cast<int>(target_shape.size());

    while (g.dim() > target_dim)
        g = g.sum(0).detach();

    for (int i = 0; i < target_dim; ++i) {
        if (target_shape[i] == 1 && g.sizes()[i] != 1) {
            g = g.sum(i).detach();
            // sum removes the dimension; unsqueeze to restore it at position i
            g = g.unsqueeze(i);
        }
    }
    return g.reshape(target_shape);
}

std::vector<Tensor> AccumulateGrad::apply(const std::vector<Tensor>& grads) {
    if (holder) {
        if (!holder->grad) {
            holder->grad = std::make_shared<Tensor>(grads[0].clone());
        } else {
            holder->grad->add_(grads[0]);
        }
    }
    return {};
}

std::vector<Tensor> NegBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    return {g.neg()};
}

std::vector<Tensor> ExpBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    return {g.mul(result)};
}

std::vector<Tensor> LogBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    return {g.div(self)};
}

std::vector<Tensor> ReluBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor mask = self.apply_unary([](float x) { return x > 0 ? 1.0f : 0.0f; });
    return {g.mul(mask)};
}

std::vector<Tensor> SigmoidBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor one_minus = result.apply_unary([](float x) { return 1.0f - x; });
    return {g.mul(result).mul(one_minus)};
}

std::vector<Tensor> TanhBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor one_minus_sq = result.apply_unary([](float x) { return 1.0f - x * x; });
    return {g.mul(one_minus_sq)};
}

std::vector<Tensor> SqrtBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor two_result = result.mul_scalar(2.0f);
    return {g.div(two_result)};
}

std::vector<Tensor> AbsBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor sign = self.apply_unary([](float x) { return x > 0 ? 1.0f : (x < 0 ? -1.0f : 0.0f); });
    return {g.mul(sign)};
}

std::vector<Tensor> AddBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor ga = reduce_grad_for_broadcast(g, self_shape);
    Tensor gb = reduce_grad_for_broadcast(g, other_shape);
    return {ga, gb};
}

std::vector<Tensor> SubBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor ga = reduce_grad_for_broadcast(g, self_shape);
    Tensor gb = reduce_grad_for_broadcast(g.neg(), other_shape);
    return {ga, gb};
}

std::vector<Tensor> MulBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor ga = reduce_grad_for_broadcast(g.mul(other), self.sizes());
    Tensor gb = reduce_grad_for_broadcast(g.mul(self), other.sizes());
    return {ga, gb};
}

std::vector<Tensor> DivBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor ga = reduce_grad_for_broadcast(g.div(other), self.sizes());
    Tensor neg_self = self.neg();
    Tensor other_sq = other.mul(other);
    Tensor gb = reduce_grad_for_broadcast(g.mul(neg_self).div(other_sq), other.sizes());
    return {ga, gb};
}

std::vector<Tensor> AddScalarBackward::apply(const std::vector<Tensor>& grads) {
    return {grads[0].detach()};
}

std::vector<Tensor> SubScalarBackward::apply(const std::vector<Tensor>& grads) {
    return {grads[0].detach()};
}

std::vector<Tensor> MulScalarBackward::apply(const std::vector<Tensor>& grads) {
    return {grads[0].detach().mul_scalar(scalar)};
}

std::vector<Tensor> DivScalarBackward::apply(const std::vector<Tensor>& grads) {
    return {grads[0].detach().div_scalar(scalar)};
}

std::vector<Tensor> PowScalarBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    return {g.mul(self.pow_scalar(scalar - 1.0f)).mul_scalar(scalar)};
}

std::vector<Tensor> MmBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor grad_self = g.mm(other.transpose(0, 1));
    Tensor grad_other = self.transpose(0, 1).mm(g);
    return {grad_self, grad_other};
}

std::vector<Tensor> BmmBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor grad_self = g.bmm(other.transpose(1, 2));
    Tensor grad_other = self.transpose(1, 2).bmm(g);
    return {grad_self, grad_other};
}

std::vector<Tensor> SumBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    if (dim == -1) {
        return {Tensor::ones(self_shape).mul_scalar(g.item())};
    }
    // g has dim removed; unsqueeze to restore it, then expand
    Tensor g_unsq = g.unsqueeze(dim);
    return {g_unsq.expand(self_shape).contiguous()};
}

std::vector<Tensor> MeanBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    float count;
    if (dim == -1) {
        count = static_cast<float>(Tensor::compute_numel(self_shape));
        return {Tensor::ones(self_shape).mul_scalar(g.item() / count)};
    }
    count = static_cast<float>(self_shape[dim]);
    Tensor g_unsq = g.unsqueeze(dim);
    return {g_unsq.expand(self_shape).contiguous().div_scalar(count)};
}

std::vector<Tensor> ReshapeBackward::apply(const std::vector<Tensor>& grads) {
    return {grads[0].detach().reshape(self_shape)};
}

std::vector<Tensor> TransposeBackward::apply(const std::vector<Tensor>& grads) {
    return {grads[0].detach().transpose(dim0, dim1).contiguous()};
}

std::vector<Tensor> ExpandBackward::apply(const std::vector<Tensor>& grads) {
    return {reduce_grad_for_broadcast(grads[0].detach(), self_shape)};
}

std::vector<Tensor> SelectBackward::apply(const std::vector<Tensor>& grads) {
    Tensor result = Tensor::zeros(self_shape);
    int n = grads[0].numel();
    Tensor gc = grads[0].contiguous();
    const float* gp = gc.data_ptr();

    std::vector<int> idx(grads[0].dim(), 0);
    for (int i = 0; i < n; ++i) {
        std::vector<int> full_idx;
        int gi = 0;
        for (int d = 0; d < static_cast<int>(self_shape.size()); ++d) {
            if (d == dim) full_idx.push_back(index);
            else full_idx.push_back(idx[gi++]);
        }
        int off = 0;
        for (int d = 0; d < static_cast<int>(self_shape.size()); ++d)
            off += full_idx[d] * result.strides()[d];
        result.data_ptr()[off] = gp[i];

        for (int d = grads[0].dim() - 1; d >= 0; --d) {
            if (++idx[d] < grads[0].sizes()[d]) break;
            idx[d] = 0;
        }
    }
    return {result};
}

std::vector<Tensor> SliceBackward::apply(const std::vector<Tensor>& grads) {
    Tensor result = Tensor::zeros(self_shape);
    Tensor dst = result.slice(dim, start, end, step);
    dst.copy_(grads[0]);
    return {result};
}

std::vector<Tensor> SoftmaxBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor sum_g = g.mul(result).sum(dim).detach();
    // sum removes the dim; unsqueeze to restore it for broadcasting
    sum_g = sum_g.unsqueeze(dim);
    Tensor expanded = sum_g.expand(result.sizes()).contiguous();
    return {result.mul(g.sub(expanded))};
}

std::vector<Tensor> LogSoftmaxBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    Tensor softmax_val = result.apply_unary([](float x) { return std::exp(x); });
    Tensor sum_g = g.sum(dim).detach();
    sum_g = sum_g.unsqueeze(dim);
    Tensor expanded = sum_g.expand(result.sizes()).contiguous();
    return {g.sub(softmax_val.mul(expanded))};
}

std::vector<Tensor> MaskedFillBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach().clone();
    int n = g.numel();
    float* gp = g.data_ptr();
    Tensor mc = mask.contiguous();
    const float* mp = mc.data_ptr();
    for (int i = 0; i < n; ++i)
        if (mp[i] != 0.0f) gp[i] = 0.0f;
    return {g};
}

std::vector<Tensor> EmbeddingBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].contiguous();
    int embed_dim = g.sizes().back();
    int n_indices = indices.numel();
    Tensor result = Tensor::zeros({num_embeddings, embed_dim});
    float* rp = result.data_ptr();
    const float* gp = g.data_ptr();
    Tensor ic = indices.contiguous();
    const float* ip = ic.data_ptr();

    for (int i = 0; i < n_indices; ++i) {
        int idx = static_cast<int>(ip[i]);
        for (int j = 0; j < embed_dim; ++j)
            rp[idx * embed_dim + j] += gp[i * embed_dim + j];
    }
    return {result};
}

std::vector<Tensor> CrossEntropyBackward::apply(const std::vector<Tensor>& grads) {
    int N = log_probs.sizes()[0];
    int C = log_probs.sizes()[1];
    Tensor softmax_val = log_probs.apply_unary([](float x) { return std::exp(x); });
    Tensor result = softmax_val.clone();
    float* rp = result.data_ptr();
    Tensor tc = targets.contiguous();
    const float* tp = tc.data_ptr();

    for (int i = 0; i < N; ++i) {
        int cls = static_cast<int>(tp[i]);
        rp[i * C + cls] -= 1.0f;
    }
    float scale = grads[0].item() / static_cast<float>(N);
    result.mul_scalar_(scale);
    return {result};
}

std::vector<Tensor> CatBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    std::vector<Tensor> result;
    int offset = 0;
    for (size_t i = 0; i < split_sizes.size(); ++i) {
        result.push_back(g.narrow(dim, offset, split_sizes[i]).contiguous());
        offset += split_sizes[i];
    }
    return result;
}

std::vector<Tensor> UnsqueezeBackward::apply(const std::vector<Tensor>& grads) {
    return {grads[0].detach().squeeze(dim)};
}

std::vector<Tensor> SqueezeBackward::apply(const std::vector<Tensor>& grads) {
    return {grads[0].detach().reshape(self_shape)};
}

std::vector<Tensor> PermuteBackward::apply(const std::vector<Tensor>& grads) {
    Tensor g = grads[0].detach();
    std::vector<int> inv(order.size());
    for (size_t i = 0; i < order.size(); ++i)
        inv[order[i]] = static_cast<int>(i);
    return {g.permute(inv).contiguous()};
}

std::vector<Tensor> GELUBackward::apply(const std::vector<Tensor>& grads) {
    static const float kSqrt2OverPi = std::sqrt(2.0f / 3.14159265358979323846f);
    int n = self.numel();
    Tensor result(self.sizes());
    Tensor sc = self.contiguous();
    const float* sp = sc.data_ptr();
    Tensor gc = grads[0].contiguous();
    const float* gp = gc.data_ptr();
    float* rp = result.data_ptr();

    for (int i = 0; i < n; ++i) {
        float x = sp[i];
        float cube = x * x * x;
        float inner = kSqrt2OverPi * (x + 0.044715f * cube);
        float tanh_val = std::tanh(inner);
        float sech2 = 1.0f - tanh_val * tanh_val;
        float d_inner = kSqrt2OverPi * (1.0f + 3.0f * 0.044715f * x * x);
        float grad_val = 0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner;
        rp[i] = gp[i] * grad_val;
    }
    return {result};
}

std::vector<Tensor> LayerNormBackward::apply(const std::vector<Tensor>& grads) {
    Tensor grad_out = grads[0].contiguous();
    int norm_dim = normalized_dim;
    int outer = self.numel() / norm_dim;

    Tensor grad_input(self.sizes());
    Tensor grad_weight = Tensor::zeros({norm_dim});
    Tensor grad_bias = Tensor::zeros({norm_dim});

    const float* go = grad_out.data_ptr();
    Tensor sc = self.contiguous();
    const float* inp = sc.data_ptr();
    const float* mp = mean.data_ptr();
    const float* rp = rstd.data_ptr();
    Tensor wc = weight.contiguous();
    const float* wp = wc.data_ptr();
    float* gi = grad_input.data_ptr();
    float* gw = grad_weight.data_ptr();
    float* gb = grad_bias.data_ptr();

    for (int i = 0; i < outer; ++i) {
        const float* go_row = go + i * norm_dim;
        const float* inp_row = inp + i * norm_dim;
        float m = mp[i];
        float rs = rp[i];
        float* gi_row = gi + i * norm_dim;

        float sum_go_w = 0.0f;
        float sum_go_w_xhat = 0.0f;
        for (int j = 0; j < norm_dim; ++j) {
            float xhat = (inp_row[j] - m) * rs;
            sum_go_w += go_row[j] * wp[j];
            sum_go_w_xhat += go_row[j] * wp[j] * xhat;
            gw[j] += go_row[j] * xhat;
            gb[j] += go_row[j];
        }

        float inv_n = 1.0f / norm_dim;
        for (int j = 0; j < norm_dim; ++j) {
            float xhat = (inp_row[j] - m) * rs;
            gi_row[j] = rs * wp[j] * (go_row[j] - inv_n * (sum_go_w + xhat * sum_go_w_xhat));
        }
    }

    return {grad_input, grad_weight, grad_bias};
}

} // namespace minitorch
