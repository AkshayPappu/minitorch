#include <gtest/gtest.h>
#include <minitorch/tensor.hpp>
#include <minitorch/nn.hpp>
#include "tokenizer.hpp"
#include "gpt.hpp"

using namespace minitorch;

static constexpr float EPS = 1e-4f;

// ── Tokenizer Tests ──

TEST(Tokenizer, BuildVocab) {
    gpt::CharTokenizer tok;
    tok.build_vocab("hello world");
    EXPECT_GT(tok.vocab_size(), 0);
    EXPECT_LE(tok.vocab_size(), 256);
}

TEST(Tokenizer, EncodeDecode) {
    gpt::CharTokenizer tok;
    tok.build_vocab("abcdef");
    auto encoded = tok.encode("abcdef");
    EXPECT_EQ(encoded.size(), 6u);
    auto decoded = tok.decode(encoded);
    EXPECT_EQ(decoded, "abcdef");
}

TEST(Tokenizer, RoundTrip) {
    gpt::CharTokenizer tok;
    std::string text = "Hello, World! 123";
    tok.build_vocab(text);
    auto encoded = tok.encode(text);
    auto decoded = tok.decode(encoded);
    EXPECT_EQ(decoded, text);
}

// ── NN Module Tests ──

TEST(NN, LinearForwardShape) {
    nn::Linear linear(10, 5);
    Tensor input = Tensor::randn({3, 10});
    Tensor output = linear.forward(input);
    EXPECT_EQ(output.sizes()[0], 3);
    EXPECT_EQ(output.sizes()[1], 5);
}

TEST(NN, Linear3DForwardShape) {
    nn::Linear linear(8, 4);
    Tensor input = Tensor::randn({2, 5, 8});
    Tensor output = linear.forward(input);
    EXPECT_EQ(output.sizes()[0], 2);
    EXPECT_EQ(output.sizes()[1], 5);
    EXPECT_EQ(output.sizes()[2], 4);
}

TEST(NN, EmbeddingForwardShape) {
    nn::Embedding emb(100, 32);
    Tensor indices({2, 5});
    for (int i = 0; i < 10; ++i)
        indices.data_ptr()[i] = static_cast<float>(i % 100);
    Tensor output = emb.forward(indices);
    EXPECT_EQ(output.sizes()[0], 2);
    EXPECT_EQ(output.sizes()[1], 5);
    EXPECT_EQ(output.sizes()[2], 32);
}

TEST(NN, LayerNormForwardShape) {
    nn::LayerNorm ln(16);
    Tensor input = Tensor::randn({2, 5, 16});
    Tensor output = ln.forward(input);
    EXPECT_EQ(output.sizes()[0], 2);
    EXPECT_EQ(output.sizes()[1], 5);
    EXPECT_EQ(output.sizes()[2], 16);
}

TEST(NN, LayerNormNormalized) {
    nn::LayerNorm ln(8);
    Tensor input = Tensor::randn({1, 1, 8});
    Tensor output = ln.forward(input);
    Tensor flat = output.reshape({8});
    float m = flat.mean().item();
    EXPECT_NEAR(m, 0.0f, 0.1f);
}

TEST(NN, DropoutTrainVsEval) {
    nn::Dropout drop(0.5f);
    Tensor input = Tensor::ones({100});

    drop.train();
    Tensor train_out = drop.forward(input);

    drop.eval();
    Tensor eval_out = drop.forward(input);
    for (int i = 0; i < 100; ++i)
        EXPECT_NEAR(eval_out.data_ptr()[i], 1.0f, EPS);
}

TEST(NN, Parameters) {
    nn::Linear linear(10, 5);
    auto params = linear.parameters();
    EXPECT_EQ(params.size(), 2u);
}

// ── GPT Model Tests ──

TEST(GPT, ForwardShape) {
    gpt::GPTConfig config;
    config.vocab_size = 26;
    config.n_embed = 32;
    config.n_heads = 4;
    config.n_layers = 1;
    config.block_size = 16;
    config.dropout = 0.0f;

    gpt::GPT model(config);
    Tensor input({1, 8});
    for (int i = 0; i < 8; ++i)
        input.set({0, i}, static_cast<float>(i % 26));

    Tensor logits = model.forward(input);
    EXPECT_EQ(logits.dim(), 3);
    EXPECT_EQ(logits.sizes()[0], 1);
    EXPECT_EQ(logits.sizes()[1], 8);
    EXPECT_EQ(logits.sizes()[2], 26);
}

TEST(GPT, ParameterCount) {
    gpt::GPTConfig config;
    config.vocab_size = 26;
    config.n_embed = 32;
    config.n_heads = 4;
    config.n_layers = 1;
    config.block_size = 16;
    config.dropout = 0.0f;

    gpt::GPT model(config);
    auto params = model.parameters();
    int total = 0;
    for (auto* p : params) total += p->numel();
    EXPECT_GT(total, 0);
    EXPECT_GT(static_cast<int>(params.size()), 10);
}

// ── New Tensor Op Tests ──

TEST(TensorOps, Softmax) {
    Tensor t({1, 4});
    t.set({0, 0}, 1.0f);
    t.set({0, 1}, 2.0f);
    t.set({0, 2}, 3.0f);
    t.set({0, 3}, 4.0f);
    Tensor s = t.softmax(1);
    float sum = 0;
    for (int i = 0; i < 4; ++i) sum += s.at({0, i});
    EXPECT_NEAR(sum, 1.0f, 1e-5f);
    EXPECT_GT(s.at({0, 3}), s.at({0, 0}));
}

TEST(TensorOps, MaskedFill) {
    Tensor t = Tensor::ones({2, 2});
    Tensor mask({2, 2});
    mask.set({0, 0}, 0.0f);
    mask.set({0, 1}, 1.0f);
    mask.set({1, 0}, 1.0f);
    mask.set({1, 1}, 0.0f);
    Tensor result = t.masked_fill(mask, -1e9f);
    EXPECT_NEAR(result.at({0, 0}), 1.0f, EPS);
    EXPECT_NEAR(result.at({0, 1}), -1e9f, 1.0f);
    EXPECT_NEAR(result.at({1, 0}), -1e9f, 1.0f);
    EXPECT_NEAR(result.at({1, 1}), 1.0f, EPS);
}

TEST(TensorOps, Randn) {
    Tensor t = Tensor::randn({100});
    float sum = 0;
    for (int i = 0; i < 100; ++i) sum += t.data_ptr()[i];
    float mean = sum / 100.0f;
    EXPECT_NEAR(mean, 0.0f, 0.5f);
}

TEST(TensorOps, Tril) {
    Tensor t = Tensor::tril(3);
    EXPECT_NEAR(t.at({0, 0}), 1.0f, EPS);
    EXPECT_NEAR(t.at({0, 1}), 0.0f, EPS);
    EXPECT_NEAR(t.at({1, 0}), 1.0f, EPS);
    EXPECT_NEAR(t.at({1, 1}), 1.0f, EPS);
    EXPECT_NEAR(t.at({2, 2}), 1.0f, EPS);
}

TEST(TensorOps, GELU) {
    Tensor t({3});
    t.data_ptr()[0] = -1.0f;
    t.data_ptr()[1] = 0.0f;
    t.data_ptr()[2] = 1.0f;
    Tensor g = t.gelu();
    EXPECT_NEAR(g.data_ptr()[1], 0.0f, 0.01f);
    EXPECT_GT(g.data_ptr()[2], 0.5f);
    EXPECT_LT(g.data_ptr()[0], 0.0f);
}

TEST(TensorOps, CrossEntropy) {
    Tensor logits({2, 3});
    logits.set({0, 0}, 2.0f); logits.set({0, 1}, 1.0f); logits.set({0, 2}, 0.1f);
    logits.set({1, 0}, 0.1f); logits.set({1, 1}, 2.0f); logits.set({1, 2}, 0.5f);
    logits.set_requires_grad(true);

    Tensor targets({2});
    targets.data_ptr()[0] = 0.0f;
    targets.data_ptr()[1] = 1.0f;

    Tensor loss = logits.cross_entropy_loss(targets);
    EXPECT_GT(loss.item(), 0.0f);
    EXPECT_LT(loss.item(), 5.0f);
}
