#pragma once

#include <vector>
#include <memory>
#include <functional>
#include <unordered_set>

namespace minitorch {

class Tensor;

struct Edge {
    std::shared_ptr<class GradFunction> function;
    int input_nr;
};

class GradFunction : public std::enable_shared_from_this<GradFunction> {
public:
    virtual ~GradFunction() = default;
    virtual std::vector<Tensor> apply(const std::vector<Tensor>& grads) = 0;

    std::vector<Edge> next_edges;

    void add_next_edge(Edge edge) {
        next_edges.push_back(std::move(edge));
    }
};

struct GradHolder {
    std::shared_ptr<Tensor> grad;
};

class AccumulateGrad : public GradFunction {
public:
    std::shared_ptr<GradHolder> holder;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

// ── Unary backward nodes ──

class NegBackward : public GradFunction {
public:
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class ExpBackward : public GradFunction {
public:
    Tensor result;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class LogBackward : public GradFunction {
public:
    Tensor self;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class ReluBackward : public GradFunction {
public:
    Tensor self;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class SigmoidBackward : public GradFunction {
public:
    Tensor result;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class TanhBackward : public GradFunction {
public:
    Tensor result;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class SqrtBackward : public GradFunction {
public:
    Tensor result;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class AbsBackward : public GradFunction {
public:
    Tensor self;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

// ── Binary backward nodes ──

class AddBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    std::vector<int> other_shape;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class SubBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    std::vector<int> other_shape;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class MulBackward : public GradFunction {
public:
    Tensor self;
    Tensor other;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class DivBackward : public GradFunction {
public:
    Tensor self;
    Tensor other;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

// ── Scalar backward nodes ──

class AddScalarBackward : public GradFunction {
public:
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class MulScalarBackward : public GradFunction {
public:
    float scalar;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class SubScalarBackward : public GradFunction {
public:
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class DivScalarBackward : public GradFunction {
public:
    float scalar;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class PowScalarBackward : public GradFunction {
public:
    Tensor self;
    float scalar;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

// ── MatMul backward ──

class MmBackward : public GradFunction {
public:
    Tensor self;
    Tensor other;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class BmmBackward : public GradFunction {
public:
    Tensor self;
    Tensor other;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

// ── Reduction backward nodes ──

class SumBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    int dim;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class MeanBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    int dim;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

// ── View / shape backward nodes ──

class ReshapeBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class TransposeBackward : public GradFunction {
public:
    int dim0, dim1;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class ExpandBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class SelectBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    int dim, index;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class SliceBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    int dim, start, end, step;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

// ── New ops for GPT ──

class SoftmaxBackward : public GradFunction {
public:
    Tensor result;
    int dim;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class LogSoftmaxBackward : public GradFunction {
public:
    Tensor result;
    int dim;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class MaskedFillBackward : public GradFunction {
public:
    Tensor mask;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class EmbeddingBackward : public GradFunction {
public:
    Tensor indices;
    int num_embeddings;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class CrossEntropyBackward : public GradFunction {
public:
    Tensor log_probs;
    Tensor targets;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class CatBackward : public GradFunction {
public:
    std::vector<int> split_sizes;
    int dim;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class UnsqueezeBackward : public GradFunction {
public:
    int dim;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class SqueezeBackward : public GradFunction {
public:
    std::vector<int> self_shape;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class PermuteBackward : public GradFunction {
public:
    std::vector<int> order;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class GELUBackward : public GradFunction {
public:
    Tensor self;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

class LayerNormBackward : public GradFunction {
public:
    Tensor self;
    Tensor mean;
    Tensor rstd;
    Tensor weight;
    int normalized_dim;
    std::vector<Tensor> apply(const std::vector<Tensor>& grads) override;
};

Tensor reduce_grad_for_broadcast(const Tensor& grad, const std::vector<int>& target_shape);

} // namespace minitorch
