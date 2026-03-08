#pragma once

#include <minitorch/tensor.hpp>
#include <vector>

namespace minitorch {
namespace optim {

class Optimizer {
public:
    explicit Optimizer(std::vector<Tensor*> params, float lr)
        : params_(std::move(params)), lr_(lr) {}
    virtual ~Optimizer() = default;

    virtual void step() = 0;
    void zero_grad();

protected:
    std::vector<Tensor*> params_;
    float lr_;
};

class SGD : public Optimizer {
public:
    SGD(std::vector<Tensor*> params, float lr, float momentum = 0.0f);
    void step() override;

private:
    float momentum_;
    std::vector<Tensor> velocity_;
};

class Adam : public Optimizer {
public:
    Adam(std::vector<Tensor*> params, float lr = 1e-3f,
         float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f);
    void step() override;

private:
    float beta1_, beta2_, eps_;
    int t_ = 0;
    std::vector<Tensor> m_;
    std::vector<Tensor> v_;
};

} // namespace optim
} // namespace minitorch
