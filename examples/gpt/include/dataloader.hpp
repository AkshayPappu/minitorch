#pragma once

#include <minitorch/tensor.hpp>
#include "tokenizer.hpp"
#include <vector>
#include <string>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gpt {

class TextDataset {
public:
    TextDataset(const std::string& filepath, CharTokenizer& tokenizer, int block_size) {
        std::ifstream file(filepath);
        if (!file.is_open())
            throw std::runtime_error("Cannot open file: " + filepath);
        std::stringstream buf;
        buf << file.rdbuf();
        text_ = buf.str();

        tokenizer.build_vocab(text_);
        tokens_ = tokenizer.encode(text_);
        block_size_ = block_size;
    }

    int size() const {
        return static_cast<int>(tokens_.size()) - block_size_;
    }

    std::pair<minitorch::Tensor, minitorch::Tensor> get(int idx) const {
        minitorch::Tensor input({block_size_});
        minitorch::Tensor target({block_size_});
        for (int i = 0; i < block_size_; ++i) {
            input.data_ptr()[i] = static_cast<float>(tokens_[idx + i]);
            target.data_ptr()[i] = static_cast<float>(tokens_[idx + i + 1]);
        }
        return {input, target};
    }

    const std::string& text() const { return text_; }

private:
    std::string text_;
    std::vector<int> tokens_;
    int block_size_;
};

class DataLoader {
public:
    DataLoader(const TextDataset& dataset, int batch_size, bool shuffle = true)
        : dataset_(dataset), batch_size_(batch_size), shuffle_(shuffle) {
        indices_.resize(dataset.size());
        for (int i = 0; i < dataset.size(); ++i)
            indices_[i] = i;
        reset();
    }

    void reset() {
        pos_ = 0;
        if (shuffle_) {
            std::shuffle(indices_.begin(), indices_.end(), rng_);
        }
    }

    bool has_next() const {
        return pos_ + batch_size_ <= static_cast<int>(indices_.size());
    }

    std::pair<minitorch::Tensor, minitorch::Tensor> next() {
        int block_size = dataset_.get(0).first.sizes()[0];
        minitorch::Tensor inputs({batch_size_, block_size});
        minitorch::Tensor targets({batch_size_, block_size});

        for (int b = 0; b < batch_size_; ++b) {
            auto [inp, tgt] = dataset_.get(indices_[pos_ + b]);
            for (int i = 0; i < block_size; ++i) {
                inputs.set({b, i}, inp.data_ptr()[i]);
                targets.set({b, i}, tgt.data_ptr()[i]);
            }
        }
        pos_ += batch_size_;
        return {inputs, targets};
    }

private:
    const TextDataset& dataset_;
    int batch_size_;
    bool shuffle_;
    int pos_ = 0;
    std::vector<int> indices_;
    std::mt19937 rng_{42};
};

} // namespace gpt
