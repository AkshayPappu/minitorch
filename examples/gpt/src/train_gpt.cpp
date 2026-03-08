#include <minitorch/tensor.hpp>
#include <minitorch/nn.hpp>
#include <minitorch/optim.hpp>
#include "gpt.hpp"
#include "tokenizer.hpp"
#include "dataloader.hpp"

#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

using namespace minitorch;

int main(int argc, char* argv[]) {
    std::string data_path = "data/input.txt";
    if (argc > 1) data_path = argv[1];

    // ── Config ──
    gpt::GPTConfig config;
    config.n_embed = 64;
    config.n_heads = 4;
    config.n_layers = 2;
    config.block_size = 32;
    config.dropout = 0.0f;

    int batch_size = 4;
    int max_steps = 200;
    float learning_rate = 3e-4f;
    int eval_interval = 50;
    int generate_interval = 100;
    int generate_tokens = 100;

    // ── Data ──
    std::cout << "Loading data from: " << data_path << std::endl;
    gpt::CharTokenizer tokenizer;
    gpt::TextDataset dataset(data_path, tokenizer, config.block_size);
    gpt::DataLoader loader(dataset, batch_size);
    config.vocab_size = tokenizer.vocab_size();

    std::cout << "Vocab size: " << config.vocab_size << std::endl;
    std::cout << "Dataset size: " << dataset.size() << " samples" << std::endl;
    std::cout << "Model: " << config.n_layers << " layers, "
              << config.n_heads << " heads, "
              << config.n_embed << " embed dim, "
              << config.block_size << " block size" << std::endl;

    // ── Model ──
    gpt::GPT model(config);
    auto params = model.parameters();
    int total_params = 0;
    for (auto* p : params) total_params += p->numel();
    std::cout << "Total parameters: " << total_params << std::endl;

    // ── Optimizer ──
    optim::Adam optimizer(params, learning_rate);

    // ── Training ──
    std::cout << "\n--- Training ---\n" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 1; step <= max_steps; ++step) {
        if (!loader.has_next()) loader.reset();
        auto [inputs, targets] = loader.next();

        model.train();
        Tensor logits = model.forward(inputs);

        int B = logits.sizes()[0];
        int T = logits.sizes()[1];
        int V = logits.sizes()[2];
        Tensor logits_flat = logits.reshape({B * T, V});
        Tensor targets_flat = targets.reshape({B * T});

        Tensor loss = logits_flat.cross_entropy_loss(targets_flat);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        float loss_val = loss.item();

        if (step % eval_interval == 0 || step == 1) {
            auto t_now = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(t_now - t_start).count();
            std::cout << "Step " << std::setw(4) << step
                      << " | Loss: " << std::fixed << std::setprecision(4) << loss_val
                      << " | Time: " << std::fixed << std::setprecision(1) << elapsed << "s"
                      << std::endl;
        }

        if (step % generate_interval == 0) {
            std::cout << "\n--- Sample generation ---" << std::endl;
            Tensor seed({1, 1}, 0.0f);
            Tensor generated = model.generate(seed, generate_tokens, 0.8f);
            std::vector<int> gen_tokens;
            for (int i = 0; i < generated.sizes()[1]; ++i)
                gen_tokens.push_back(static_cast<int>(generated.at({0, i})));
            std::cout << tokenizer.decode(gen_tokens) << std::endl;
            std::cout << "---\n" << std::endl;
        }
    }

    // ── Final generation ──
    std::cout << "\n=== Final Generation ===" << std::endl;
    Tensor seed({1, 1}, 0.0f);
    Tensor generated = model.generate(seed, 200, 0.8f);
    std::vector<int> gen_tokens;
    for (int i = 0; i < generated.sizes()[1]; ++i)
        gen_tokens.push_back(static_cast<int>(generated.at({0, i})));
    std::cout << tokenizer.decode(gen_tokens) << std::endl;

    return 0;
}
