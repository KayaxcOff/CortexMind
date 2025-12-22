//
// Created by muham on 13.12.2025.
//

#include "CortexMind/tools/Tokenizer/token.hpp"
#include <sstream>

using namespace cortex::tools;

TokenNet::TokenNet() : nextIdx(4) {
    this->tokensIdx[PAD_TOKEN] = PAD_ID;
    this->tokensIdx[UNK_TOKEN] = UNK_ID;
    this->tokensIdx[BOS_TOKEN] = BOS_ID;
    this->tokensIdx[EOS_TOKEN] = EOS_ID;

    this->reverseIdx[PAD_ID] = PAD_TOKEN;
    this->reverseIdx[UNK_ID] = UNK_TOKEN;
    this->reverseIdx[BOS_ID] = BOS_TOKEN;
    this->reverseIdx[EOS_ID] = EOS_TOKEN;
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
    for (const std::string &text : tokens) {
        this->fit(text);
    }
}

std::vector<int> TokenNet::tokenize(const std::string &token, const bool add_special_tokens) {
    std::vector<int> ids;

    if (add_special_tokens) {
        ids.push_back(BOS_ID);
    }

    std::istringstream iss(token);
    std::string word;

    while (iss >> word) {

        auto it = this->tokensIdx.find(word);
        ids.push_back(it != this->tokensIdx.end() ? it->second : UNK_ID);
    }

    if (add_special_tokens) {
        ids.push_back(EOS_ID);
    }

    return ids;
}

std::vector<int> TokenNet::encode(const std::string &text, const int max_length, const bool add_special_tokens, const bool truncate) {
    std::vector<int> ids = this->tokenize(text, add_special_tokens);

    if (truncate && static_cast<int>(ids.size()) > max_length) {
        ids.resize(max_length);
        if (add_special_tokens && max_length > 0) {
            ids[max_length - 1] = EOS_ID;
        }
    } else if (static_cast<int>(ids.size()) < max_length) {
        // Pad with PAD_ID
        ids.resize(max_length, PAD_ID);
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

std::string TokenNet::decode(const std::vector<int>& ids, const bool skip_special_tokens) const {
    std::string result;

    for (const int id : ids) {
        if (skip_special_tokens && (id == PAD_ID || id == BOS_ID || id == EOS_ID)) {
            continue;
        }

        if (id == PAD_ID) break;

        std::string token = this->getToken(id);
        if (!result.empty() && token != UNK_TOKEN) {
            result += " ";
        }
        result += token;
    }

    return result;
}

size_t TokenNet::vocab_size() const {
    return this->tokensIdx.size();
}

size_t TokenNet::size() const {
    return this->vocab_size();
}