//
// Created by muham on 13.12.2025.
//

#include "CortexMind/tools/Tokenizer/token.hpp"
#include <sstream>

using namespace cortex::tools;

TokenNet::TokenNet() : nextIdx(4) {
    this->tokensIdx[this->PAD_TOKEN] = this->PAD_ID;
    this->tokensIdx[this->UNK_TOKEN] = this->UNK_ID;
    this->tokensIdx[this->BOS_TOKEN] = this->BOS_ID;
    this->tokensIdx[this->EOS_TOKEN] = this->EOS_ID;

    this->reverseIdx[this->PAD_ID] = this->PAD_TOKEN;
    this->reverseIdx[this->UNK_ID] = this->UNK_TOKEN;
    this->reverseIdx[this->BOS_ID] = this->BOS_TOKEN;
    this->reverseIdx[this->EOS_ID] = this->EOS_TOKEN;
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
        ids.push_back(this->BOS_ID);
    }

    std::istringstream iss(token);
    std::string word;

    while (iss >> word) {

        auto it = this->tokensIdx.find(word);
        ids.push_back(it != this->tokensIdx.end() ? it->second : this->UNK_ID);
    }

    if (add_special_tokens) {
        ids.push_back(this->EOS_ID);
    }

    return ids;
}

std::vector<int> TokenNet::encode(const std::string &text, const int max_length, const bool add_special_tokens, const bool truncate) {
    std::vector<int> ids = this->tokenize(text, add_special_tokens);

    if (truncate && static_cast<int>(ids.size()) > max_length) {
        ids.resize(max_length);
        if (add_special_tokens && max_length > 0) {
            ids[max_length - 1] = this->EOS_ID;
        }
    } else if (static_cast<int>(ids.size()) < max_length) {
        // Pad with PAD_ID
        ids.resize(max_length, this->PAD_ID);
    }

    return ids;
}

int TokenNet::getId(const std::string& token) const {
    const auto it = this->tokensIdx.find(token);
    return it != this->tokensIdx.end() ? it->second : this->UNK_ID;
}

std::string TokenNet::getToken(const int id) const {
    const auto it = this->reverseIdx.find(id);
    return it != this->reverseIdx.end() ? it->second : this->UNK_TOKEN;
}

std::string TokenNet::decode(const std::vector<int>& ids, const bool skip_special_tokens) const {
    std::string result;

    for (const int id : ids) {
        if (skip_special_tokens && (id == this->PAD_ID || id == this->BOS_ID || id == this->EOS_ID)) {
            continue;
        }

        if (id == this->PAD_ID) break;

        std::string token = this->getToken(id);
        if (!result.empty() && token != this->UNK_TOKEN) {
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