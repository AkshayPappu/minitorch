#pragma once

#include <minitorch/tensor.hpp>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

namespace minitorch {
namespace nn {

class Module {
public:
    virtual ~Module() = default;
    Module() = default;
    Module(const Module&) = delete;
    Module& operator=(const Module&) = delete;
    Module(Module&&) = default;
    Module& operator=(Module&&) = default;

    virtual Tensor forward(const Tensor& input) = 0;

    std::vector<Tensor*> parameters();
    void zero_grad();
    void train();
    void eval();
    bool is_training() const { return training_; }

    void register_parameter(const std::string& name, Tensor& param);
    void register_module(const std::string& name, std::shared_ptr<Module> module);

protected:
    bool training_ = true;
    std::vector<std::pair<std::string, Tensor*>> params_;
    std::vector<std::pair<std::string, std::shared_ptr<Module>>> children_;
};

class Linear : public Module {
public:
    Linear(int in_features, int out_features, bool bias = true);
    Tensor forward(const Tensor& input) override;

    Tensor weight;
    Tensor bias;
    bool has_bias;
    int in_features_, out_features_;
};

class Embedding : public Module {
public:
    Embedding(int num_embeddings, int embedding_dim);
    Tensor forward(const Tensor& input) override;

    Tensor weight;
    int num_embeddings_, embedding_dim_;
};

class LayerNorm : public Module {
public:
    LayerNorm(int normalized_shape, float eps = 1e-5f);
    Tensor forward(const Tensor& input) override;

    Tensor weight;
    Tensor bias;
    int normalized_shape_;
    float eps_;
};

class Dropout : public Module {
public:
    explicit Dropout(float p = 0.1f);
    Tensor forward(const Tensor& input) override;

    float p_;
};

class Sequential : public Module {
public:
    Sequential() = default;
    void add(std::shared_ptr<Module> module);
    Tensor forward(const Tensor& input) override;
};

} // namespace nn
} // namespace minitorch
