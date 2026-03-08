#include <minitorch/optim.hpp>
#include <cmath>

namespace minitorch {
namespace optim {

void Optimizer::zero_grad() {
    for (auto* p : params_)
        p->zero_grad();
}

// ── SGD ──

SGD::SGD(std::vector<Tensor*> params, float lr, float momentum)
    : Optimizer(std::move(params), lr), momentum_(momentum) {
    if (momentum_ > 0.0f) {
        for (auto* p : params_)
            velocity_.push_back(Tensor::zeros(p->sizes()));
    }
}

void SGD::step() {
    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor* p = params_[i];
        Tensor g = p->grad();
        if (g.numel() == 0) continue;

        if (momentum_ > 0.0f) {
            velocity_[i] = velocity_[i].mul_scalar(momentum_).add(g);
            p->add_(velocity_[i].mul_scalar(-lr_));
        } else {
            p->add_(g.mul_scalar(-lr_));
        }
    }
}

// ── Adam ──

Adam::Adam(std::vector<Tensor*> params, float lr, float beta1, float beta2, float eps)
    : Optimizer(std::move(params), lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (auto* p : params_) {
        m_.push_back(Tensor::zeros(p->sizes()));
        v_.push_back(Tensor::zeros(p->sizes()));
    }
}

void Adam::step() {
    t_++;
    float bc1 = 1.0f - std::pow(beta1_, static_cast<float>(t_));
    float bc2 = 1.0f - std::pow(beta2_, static_cast<float>(t_));

    for (size_t i = 0; i < params_.size(); ++i) {
        Tensor* p = params_[i];
        Tensor g = p->grad();
        if (g.numel() == 0) continue;

        m_[i] = m_[i].mul_scalar(beta1_).add(g.mul_scalar(1.0f - beta1_));
        v_[i] = v_[i].mul_scalar(beta2_).add(g.mul(g).mul_scalar(1.0f - beta2_));

        Tensor m_hat = m_[i].div_scalar(bc1);
        Tensor v_hat = v_[i].div_scalar(bc2);

        Tensor update = m_hat.div(v_hat.sqrt().add_scalar(eps_)).mul_scalar(-lr_);
        p->add_(update);
    }
}

} // namespace optim
} // namespace minitorch
