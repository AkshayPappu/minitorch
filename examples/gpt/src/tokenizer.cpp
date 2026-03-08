#include "tokenizer.hpp"
#include <set>
#include <stdexcept>

namespace gpt {

void CharTokenizer::build_vocab(const std::string& text) {
    std::set<char> chars(text.begin(), text.end());
    itos_.clear();
    stoi_.clear();
    for (char c : chars) {
        int idx = static_cast<int>(itos_.size());
        stoi_[c] = idx;
        itos_.push_back(c);
    }
}

std::vector<int> CharTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size());
    for (char c : text) {
        auto it = stoi_.find(c);
        if (it == stoi_.end())
            throw std::runtime_error(std::string("Unknown character: ") + c);
        tokens.push_back(it->second);
    }
    return tokens;
}

std::string CharTokenizer::decode(const std::vector<int>& tokens) const {
    std::string text;
    text.reserve(tokens.size());
    for (int t : tokens) {
        if (t < 0 || t >= static_cast<int>(itos_.size()))
            throw std::out_of_range("Token index out of range");
        text += itos_[t];
    }
    return text;
}

} // namespace gpt
