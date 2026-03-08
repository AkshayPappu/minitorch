#include "gpt.hpp"
#include <cmath>
#include <random>

using minitorch::Tensor;

namespace gpt {

// ── CausalSelfAttention ──

CausalSelfAttention::CausalSelfAttention(const GPTConfig& config)
    : config_(config) {
    qkv_proj_ = std::make_shared<minitorch::nn::Linear>(config.n_embed, 3 * config.n_embed);
    out_proj_ = std::make_shared<minitorch::nn::Linear>(config.n_embed, config.n_embed);
    attn_dropout_ = std::make_shared<minitorch::nn::Dropout>(config.dropout);
    resid_dropout_ = std::make_shared<minitorch::nn::Dropout>(config.dropout);
    register_module("qkv_proj", qkv_proj_);
    register_module("out_proj", out_proj_);
    register_module("attn_dropout", attn_dropout_);
    register_module("resid_dropout", resid_dropout_);
}

Tensor CausalSelfAttention::forward(const Tensor& input) {
    int B = input.sizes()[0];
    int T = input.sizes()[1];
    int C = input.sizes()[2];
    int n_heads = config_.n_heads;
    int head_dim = C / n_heads;

    Tensor qkv = qkv_proj_->forward(input);

    Tensor q = qkv.slice(2, 0, C, 1).contiguous();
    Tensor k = qkv.slice(2, C, 2 * C, 1).contiguous();
    Tensor v = qkv.slice(2, 2 * C, 3 * C, 1).contiguous();

    q = q.reshape({B, T, n_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().reshape({B * n_heads, T, head_dim});
    k = k.reshape({B, T, n_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().reshape({B * n_heads, T, head_dim});
    v = v.reshape({B, T, n_heads, head_dim}).permute({0, 2, 1, 3}).contiguous().reshape({B * n_heads, T, head_dim});

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    Tensor att = q.bmm(k.transpose(1, 2)).mul_scalar(scale);

    Tensor mask = Tensor::tril(T);
    Tensor causal_mask = mask.eq(Tensor::zeros({T, T}));
    Tensor mask_expanded = causal_mask.unsqueeze(0).expand({B * n_heads, T, T});
    att = att.masked_fill(mask_expanded, -1e9f);

    att = att.softmax(2);
    att = attn_dropout_->forward(att);

    Tensor y = att.bmm(v);
    y = y.reshape({B, n_heads, T, head_dim}).permute({0, 2, 1, 3}).contiguous().reshape({B, T, C});

    y = out_proj_->forward(y);
    y = resid_dropout_->forward(y);
    return y;
}

// ── MLP ──

MLP::MLP(const GPTConfig& config) {
    fc1_ = std::make_shared<minitorch::nn::Linear>(config.n_embed, 4 * config.n_embed);
    fc2_ = std::make_shared<minitorch::nn::Linear>(4 * config.n_embed, config.n_embed);
    dropout_ = std::make_shared<minitorch::nn::Dropout>(config.dropout);
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
    register_module("dropout", dropout_);
}

Tensor MLP::forward(const Tensor& input) {
    Tensor x = fc1_->forward(input);
    x = x.gelu();
    x = fc2_->forward(x);
    x = dropout_->forward(x);
    return x;
}

// ── TransformerBlock ──

TransformerBlock::TransformerBlock(const GPTConfig& config) {
    ln1_ = std::make_shared<minitorch::nn::LayerNorm>(config.n_embed);
    attn_ = std::make_shared<CausalSelfAttention>(config);
    ln2_ = std::make_shared<minitorch::nn::LayerNorm>(config.n_embed);
    mlp_ = std::make_shared<MLP>(config);
    register_module("ln1", ln1_);
    register_module("attn", attn_);
    register_module("ln2", ln2_);
    register_module("mlp", mlp_);
}

Tensor TransformerBlock::forward(const Tensor& input) {
    Tensor x = input.add(attn_->forward(ln1_->forward(input)));
    x = x.add(mlp_->forward(ln2_->forward(x)));
    return x;
}

// ── GPT ──

GPT::GPT(const GPTConfig& config)
    : config_(config) {
    tok_emb_ = std::make_shared<minitorch::nn::Embedding>(config.vocab_size, config.n_embed);
    pos_emb_ = std::make_shared<minitorch::nn::Embedding>(config.block_size, config.n_embed);
    drop_ = std::make_shared<minitorch::nn::Dropout>(config.dropout);
    ln_f_ = std::make_shared<minitorch::nn::LayerNorm>(config.n_embed);
    lm_head_ = std::make_shared<minitorch::nn::Linear>(config.n_embed, config.vocab_size, false);

    register_module("tok_emb", tok_emb_);
    register_module("pos_emb", pos_emb_);
    register_module("drop", drop_);

    for (int i = 0; i < config.n_layers; ++i) {
        auto block = std::make_shared<TransformerBlock>(config);
        blocks_.push_back(block);
        register_module("block_" + std::to_string(i), block);
    }

    register_module("ln_f", ln_f_);
    register_module("lm_head", lm_head_);
}

Tensor GPT::forward(const Tensor& input) {
    int B = input.sizes()[0];
    int T = input.sizes()[1];

    Tensor pos = Tensor::arange(0, static_cast<float>(T), 1.0f);

    Tensor tok = tok_emb_->forward(input);
    Tensor position = pos_emb_->forward(pos);

    Tensor pos_expanded = position.unsqueeze(0).expand({B, T, config_.n_embed});
    Tensor x = tok.add(pos_expanded);
    x = drop_->forward(x);

    for (auto& block : blocks_)
        x = block->forward(x);

    x = ln_f_->forward(x);
    Tensor logits = lm_head_->forward(x);
    return logits;
}

Tensor GPT::generate(const Tensor& context, int max_new_tokens, float temperature) {
    static std::mt19937 gen(42);
    eval();

    std::vector<int> tokens;
    Tensor ctx = context.contiguous();
    int T = ctx.sizes()[1];
    for (int i = 0; i < T; ++i)
        tokens.push_back(static_cast<int>(ctx.at({0, i})));

    for (int step = 0; step < max_new_tokens; ++step) {
        int ctx_len = std::min(static_cast<int>(tokens.size()), config_.block_size);
        int start = static_cast<int>(tokens.size()) - ctx_len;

        Tensor input({1, ctx_len});
        for (int i = 0; i < ctx_len; ++i)
            input.set({0, i}, static_cast<float>(tokens[start + i]));

        Tensor logits = forward(input);

        int last_t = ctx_len - 1;
        Tensor last_logits({1, config_.vocab_size});
        for (int j = 0; j < config_.vocab_size; ++j)
            last_logits.set({0, j}, logits.at({0, last_t, j}));

        if (temperature != 1.0f)
            last_logits = last_logits.div_scalar(temperature);

        Tensor probs = last_logits.softmax(1);

        std::vector<float> prob_vec(config_.vocab_size);
        for (int j = 0; j < config_.vocab_size; ++j)
            prob_vec[j] = probs.at({0, j});

        std::discrete_distribution<int> dist(prob_vec.begin(), prob_vec.end());
        int next_token = dist(gen);
        tokens.push_back(next_token);
    }

    Tensor result({1, static_cast<int>(tokens.size())});
    for (size_t i = 0; i < tokens.size(); ++i)
        result.set({0, static_cast<int>(i)}, static_cast<float>(tokens[i]));
    return result;
}

} // namespace gpt
