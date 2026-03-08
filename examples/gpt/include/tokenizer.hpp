#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace gpt {

class CharTokenizer {
public:
    void build_vocab(const std::string& text);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;
    int vocab_size() const { return static_cast<int>(itos_.size()); }

private:
    std::unordered_map<char, int> stoi_;
    std::vector<char> itos_;
};

} // namespace gpt
