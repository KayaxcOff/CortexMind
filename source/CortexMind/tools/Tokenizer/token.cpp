//
// Created by muham on 13.12.2025.
//

#include "CortexMind/tools/Tokenizer/token.hpp"
#include <sstream>

using namespace cortex::tools;

TokenNet::TokenNet() : nextIdx(1) {
    this->tokensIdx[UNK_TOKEN] = UNK_ID;
    this->reverseIdx[UNK_ID] = UNK_TOKEN;
}

TokenNet::~TokenNet() = default;

void TokenNet::fit(const std::string &token) {
    std::istringstream ss(token);
    std::string word;

    while (ss >> word) {
        if (!this->tokensIdx.contains(word)) {
            const int idx = this->nextIdx++;
            this->tokensIdx[word] = idx;
            this->reverseIdx[idx] = word;
        }
    }
}

void TokenNet::fit(const std::vector<std::string> &tokens) {
    for (const std::string &token : tokens) {
        this->fit(token);
    }
}

std::vector<int> TokenNet::tokenize(const std::string &token) {
    std::vector<int> ids;
    std::istringstream iss(token);
    std::string word;

    while (iss >> word) {
        auto it = this->tokensIdx.find(word);
        ids.push_back(it != this->tokensIdx.end() ? it->second : UNK_ID);
    }
    return ids;
}

int TokenNet::getId(const std::string& token) const {
    const auto it = this->tokensIdx.find(token);
    return it != this->tokensIdx.end() ? it->second : UNK_ID;
}

std::string TokenNet::getToken(const int id) const {
    const auto it = this->reverseIdx.find(id);
    return it != this->reverseIdx.end() ? it->second : UNK_TOKEN;
}

size_t TokenNet::size() const {
    return this->tokensIdx.size();
}