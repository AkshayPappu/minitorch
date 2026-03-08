#include <minitorch/nn.hpp>
#include <cmath>

namespace minitorch {
namespace nn {

// ── Module ──

std::vector<Tensor*> Module::parameters() {
    std::vector<Tensor*> result;
    for (auto& [name, param] : params_)
        result.push_back(param);
    for (auto& [name, child] : children_) {
        auto child_params = child->parameters();
        result.insert(result.end(), child_params.begin(), child_params.end());
    }
    return result;
}

void Module::zero_grad() {
    for (auto* p : parameters())
        p->zero_grad();
}

void Module::train() {
    training_ = true;
    for (auto& [name, child] : children_)
        child->train();
}

void Module::eval() {
    training_ = false;
    for (auto& [name, child] : children_)
        child->eval();
}

void Module::register_parameter(const std::string& name, Tensor& param) {
    params_.emplace_back(name, &param);
}

void Module::register_module(const std::string& name, std::shared_ptr<Module> module) {
    children_.emplace_back(name, std::move(module));
}

// ── Linear ──

Linear::Linear(int in_features, int out_features, bool use_bias)
    : has_bias(use_bias), in_features_(in_features), out_features_(out_features) {
    float k = 1.0f / std::sqrt(static_cast<float>(in_features));
    weight = Tensor::randn({out_features, in_features}).mul_scalar(k);
    weight.set_requires_grad(true);
    register_parameter("weight", weight);

    if (has_bias) {
        bias = Tensor::randn({out_features}).mul_scalar(k);
        bias.set_requires_grad(true);
        register_parameter("bias", bias);
    }
}

Tensor Linear::forward(const Tensor& input) {
    // input: (..., in_features) -> (..., out_features)
    // y = x @ W^T + b
    int ndim = input.dim();
    if (ndim == 2) {
        Tensor result = input.mm(weight.transpose(0, 1));
        if (has_bias)
            result = result.add(bias.unsqueeze(0).expand(result.sizes()));
        return result;
    }

    // For higher dims, reshape to 2D, compute, reshape back
    std::vector<int> orig_shape = input.sizes();
    int batch = 1;
    for (int i = 0; i < ndim - 1; ++i) batch *= orig_shape[i];

    Tensor flat = input.reshape({batch, in_features_});
    Tensor result = flat.mm(weight.transpose(0, 1));
    if (has_bias)
        result = result.add(bias.unsqueeze(0).expand(result.sizes()));

    std::vector<int> out_shape(orig_shape.begin(), orig_shape.end() - 1);
    out_shape.push_back(out_features_);
    return result.reshape(out_shape);
}

// ── Embedding ──

Embedding::Embedding(int num_embeddings, int embedding_dim)
    : num_embeddings_(num_embeddings), embedding_dim_(embedding_dim) {
    weight = Tensor::randn({num_embeddings, embedding_dim});
    weight.set_requires_grad(true);
    register_parameter("weight", weight);
}

Tensor Embedding::forward(const Tensor& input) {
    return weight.embedding_lookup(input);
}

// ── LayerNorm ──

LayerNorm::LayerNorm(int normalized_shape, float eps)
    : normalized_shape_(normalized_shape), eps_(eps) {
    weight = Tensor::ones({normalized_shape});
    weight.set_requires_grad(true);
    bias = Tensor::zeros({normalized_shape});
    bias.set_requires_grad(true);
    register_parameter("weight", weight);
    register_parameter("bias", bias);
}

Tensor LayerNorm::forward(const Tensor& input) {
    return input.layer_norm(weight, bias, eps_);
}

// ── Dropout ──

Dropout::Dropout(float p) : p_(p) {}

Tensor Dropout::forward(const Tensor& input) {
    return input.dropout(p_, training_);
}

// ── Sequential ──

void Sequential::add(std::shared_ptr<Module> module) {
    std::string name = "module_" + std::to_string(children_.size());
    register_module(name, std::move(module));
}

Tensor Sequential::forward(const Tensor& input) {
    Tensor x = input;
    for (auto& [name, child] : children_)
        x = child->forward(x);
    return x;
}

} // namespace nn
} // namespace minitorch
