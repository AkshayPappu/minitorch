#pragma once

#include <minitorch/tensor.hpp>
#include <minitorch/nn.hpp>
#include <memory>
#include <vector>

namespace gpt {

struct GPTConfig {
    int vocab_size = 65;
    int n_embed = 256;
    int n_heads = 8;
    int n_layers = 6;
    int block_size = 128;
    float dropout = 0.1f;
};

class CausalSelfAttention : public minitorch::nn::Module {
public:
    CausalSelfAttention(const GPTConfig& config);
    minitorch::Tensor forward(const minitorch::Tensor& input) override;

private:
    GPTConfig config_;
    std::shared_ptr<minitorch::nn::Linear> qkv_proj_;
    std::shared_ptr<minitorch::nn::Linear> out_proj_;
    std::shared_ptr<minitorch::nn::Dropout> attn_dropout_;
    std::shared_ptr<minitorch::nn::Dropout> resid_dropout_;
};

class MLP : public minitorch::nn::Module {
public:
    MLP(const GPTConfig& config);
    minitorch::Tensor forward(const minitorch::Tensor& input) override;

private:
    std::shared_ptr<minitorch::nn::Linear> fc1_;
    std::shared_ptr<minitorch::nn::Linear> fc2_;
    std::shared_ptr<minitorch::nn::Dropout> dropout_;
};

class TransformerBlock : public minitorch::nn::Module {
public:
    TransformerBlock(const GPTConfig& config);
    minitorch::Tensor forward(const minitorch::Tensor& input) override;

private:
    std::shared_ptr<minitorch::nn::LayerNorm> ln1_;
    std::shared_ptr<CausalSelfAttention> attn_;
    std::shared_ptr<minitorch::nn::LayerNorm> ln2_;
    std::shared_ptr<MLP> mlp_;
};

class GPT : public minitorch::nn::Module {
public:
    GPT(const GPTConfig& config);
    minitorch::Tensor forward(const minitorch::Tensor& input) override;
    minitorch::Tensor generate(const minitorch::Tensor& context, int max_new_tokens, float temperature = 1.0f);

    GPTConfig config_;

private:
    std::shared_ptr<minitorch::nn::Embedding> tok_emb_;
    std::shared_ptr<minitorch::nn::Embedding> pos_emb_;
    std::shared_ptr<minitorch::nn::Dropout> drop_;
    std::vector<std::shared_ptr<TransformerBlock>> blocks_;
    std::shared_ptr<minitorch::nn::LayerNorm> ln_f_;
    std::shared_ptr<minitorch::nn::Linear> lm_head_;
};

} // namespace gpt
